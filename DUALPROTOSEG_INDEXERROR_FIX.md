# DualProtoSeg IndexError Fix

## Problem

After updating the evaluation metrics in `text_seg_eval.py` to focus on foreground, DualProtoSeg encountered an IndexError:

```python
IndexError: boolean index did not match indexed array along axis 0; 
size of axis is 224 but size of corresponding boolean axis is 522

File "/home/roba/miccai26/wrappers/dualprotoseg_wrapper.py", line 623
    confidence = float(cam_np[binary_mask > 0].mean())
```

## Root Cause

The issue was caused by a **dimension mismatch** when calculating confidence:

1. **CAM output** (`cam_np`): 224×224 (model's native resolution)
2. **Binary mask** (after thresholding): 224×224 (same as CAM)
3. **Binary mask** (after resizing): 522×? (resized to original image size)
4. **Confidence calculation**: Tried to use resized mask (522×?) to index CAM (224×224)

### Code Flow (Before Fix)

```python
# Step 1: CAM is 224×224
cam_np = class_cam.cpu().numpy()  # Shape: [224, 224]

# Step 2: Binary mask is 224×224
binary_mask = (cam_np > self.threshold).astype(np.uint8)  # Shape: [224, 224]

# Step 3: Binary mask resized to original image size
binary_mask = resize(binary_mask, orig_size)  # Shape: [522, ?]

# Step 4: ERROR! Trying to index 224×224 with 522×? mask
confidence = float(cam_np[binary_mask > 0].mean())  # ❌ IndexError
```

## Solution

**Calculate confidence BEFORE resizing the mask**, when both arrays have the same dimensions:

### Code Flow (After Fix)

```python
# Step 1: CAM is 224×224
cam_np = class_cam.cpu().numpy()  # Shape: [224, 224]

# Step 2: Binary mask is 224×224
binary_mask = (cam_np > self.threshold).astype(np.uint8)  # Shape: [224, 224]

# Step 3: Calculate confidence NOW (both are 224×224)
confidence = float(cam_np[binary_mask > 0].mean())  # ✅ Works!

# Step 4: Resize binary mask to original image size
binary_mask = resize(binary_mask, orig_size)  # Shape: [522, ?]
```

## Implementation

**File**: `wrappers/dualprotoseg_wrapper.py`

**Before (Line ~613-623)**:
```python
# Apply threshold
binary_mask = (cam_np > self.threshold).astype(np.uint8)

# Resize to original size
if binary_mask.shape != (orig_size[1], orig_size[0]):
    mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
    mask_pil = mask_pil.resize(orig_size, Image.NEAREST)
    binary_mask = (np.array(mask_pil) > 127).astype(np.uint8)

# Calculate confidence from CAM values ❌ WRONG ORDER
confidence = float(cam_np[binary_mask > 0].mean()) if binary_mask.sum() > 0 else 0.0
```

**After**:
```python
# Apply threshold
binary_mask = (cam_np > self.threshold).astype(np.uint8)

# Calculate confidence from CAM values BEFORE resizing ✅ CORRECT
# (cam_np and binary_mask are same size here: 224x224)
confidence = float(cam_np[binary_mask > 0].mean()) if binary_mask.sum() > 0 else 0.0

# Resize to original size
if binary_mask.shape != (orig_size[1], orig_size[0]):
    mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
    mask_pil = mask_pil.resize(orig_size, Image.NEAREST)
    binary_mask = (np.array(mask_pil) > 127).astype(np.uint8)
```

## Why This Fix is Correct

1. **Semantic Correctness**: 
   - Confidence represents the average CAM activation within the predicted region
   - It makes more sense to calculate this at the model's native resolution (224×224)
   - The CAM values are meaningful at 224×224, not after nearest-neighbor interpolation

2. **Numerical Stability**:
   - CAM values at 224×224 are the direct output from the model
   - After resizing, we'd need to resize CAM too, introducing interpolation artifacts
   - Better to use original model outputs for confidence

3. **Efficiency**:
   - Only need to resize the binary mask, not the continuous CAM
   - Confidence calculation on 224×224 is faster than on larger sizes

## Impact on Results

This fix does NOT change the segmentation masks, only fixes the confidence calculation:

- ✅ **Segmentation masks**: Unchanged (still resized to original image size)
- ✅ **Confidence values**: Now correctly computed from 224×224 CAM
- ✅ **Metrics (IoU, Dice, etc.)**: Unchanged (computed on binary masks)

## Testing

### Verify Fix
```bash
# Should load without errors
./run.sh --list

# Run DualProtoSeg evaluation (on GPU)
srun --gres=gpu:1 --mem=32G bash run.sh dualprotoseg
```

### Expected Behavior
- ✅ No IndexError
- ✅ Confidence values computed correctly
- ✅ Segmentation masks generated at original image resolution
- ✅ Evaluation metrics computed on foreground pixels

## Related Context

This fix was needed after updating `text_seg_eval.py` to focus evaluation metrics on foreground pixels only. The evaluation changes were:

1. Using `sklearn` metrics with `pos_label=1` (foreground only)
2. IoU, Dice, Precision, Recall computed for foreground class
3. Accuracy computed on union of predicted and GT foreground
4. Added boundary IoU and Hausdorff distance

These evaluation improvements revealed the dimension mismatch bug in DualProtoSeg's confidence calculation.

## Files Modified

✅ **`wrappers/dualprotoseg_wrapper.py`**
- Moved confidence calculation before mask resizing
- Added clarifying comment about dimension matching

## Summary

✅ **IndexError** - FIXED (moved confidence calculation)  
✅ **Dimension mismatch** - RESOLVED (calculate before resize)  
✅ **Semantic correctness** - IMPROVED (use native CAM resolution)  
✅ **Confidence values** - NOW CORRECT  

DualProtoSeg now works correctly with the foreground-focused evaluation metrics!
