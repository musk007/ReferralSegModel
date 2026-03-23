# DataLoader Key Error - FIXED

## Problem

When running evaluation, a KeyError occurred:

```python
KeyError: 'image'
  File "/home/roba/miccai26/text_seg_eval.py", line 141, in evaluate_model
    image = sample["image"]
```

## Root Cause

**Mismatch between dataset structure and evaluation code:**

**Dataset (in `text_seg_eval_example.py`)** stores:
- `"image_path"` - Path to image file
- `"mask_path"` - Path to mask file
- `"prompt"` - Text prompt
- `"id"` - Sample identifier

**Evaluation code (in `text_seg_eval.py`)** expected:
- `"image"` - Loaded PIL Image
- `"mask"` - Loaded numpy array
- `"text_prompt"` - Text prompt

## Solution

Updated `text_seg_eval.py` to handle both formats:

### Before (Broken)
```python
image = sample["image"]
gt_mask = sample["mask"]
text_prompt = sample["text_prompt"]
```

### After (Fixed)
```python
# Load image from path if needed
if "image" in sample:
    image = sample["image"]
elif "image_path" in sample:
    image = Image.open(sample["image_path"]).convert("RGB")
else:
    raise KeyError("Sample must contain either 'image' or 'image_path'")

# Load mask from path if needed
if "mask" in sample:
    gt_mask = sample["mask"]
elif "mask_path" in sample:
    gt_mask = np.array(Image.open(sample["mask_path"]))
else:
    raise KeyError("Sample must contain either 'mask' or 'mask_path'")

# Handle both prompt keys
text_prompt = sample.get("prompt") or sample.get("text_prompt", "")
```

### Additional Improvement

Updated prediction saving to use sample ID:

```python
# Before
pred_img.save(os.path.join(save_predictions, f"sample_{i:04d}_pred.png"))

# After  
sample_id = sample.get("id", f"sample_{i:04d}")
pred_img.save(os.path.join(save_predictions, f"{sample_id}_pred.png"))
```

## Benefits

1. ✅ **Flexible**: Supports both path-based and pre-loaded datasets
2. ✅ **Memory Efficient**: Loads images on-demand instead of all at once
3. ✅ **Better Naming**: Uses meaningful sample IDs in saved predictions
4. ✅ **Backward Compatible**: Still works with old format if provided

## Implementation Details

### Path-Based Loading (Current)
```python
dataset.append({
    "id": img_name,
    "image_path": img_path,
    "mask_path": mask_path,
    "prompt": prompt,
})
```

**Advantages:**
- Lower memory usage (images loaded on-demand)
- Faster dataset loading
- Better for large datasets

### Pre-Loaded Format (Also Supported)
```python
dataset.append({
    "image": PIL.Image.open(path),
    "mask": np.array(PIL.Image.open(mask_path)),
    "text_prompt": prompt,
})
```

**Advantages:**
- Faster inference (no I/O during evaluation)
- Good for small datasets
- Useful when preprocessing is needed

## Files Modified

✅ **`text_seg_eval.py`**
- Added flexible image/mask loading
- Handle both `prompt` and `text_prompt` keys
- Use sample ID for prediction filenames

## Testing

### Verify Fix
```bash
# Should load successfully without KeyError
./run.sh --list

# Run evaluation (on GPU)
srun --gres=gpu:1 --mem=32G bash run.sh biomedparse
```

### Expected Behavior
- ✅ No KeyError
- ✅ Images loaded correctly
- ✅ Predictions saved with meaningful names (e.g., `image001_pred.png` instead of `sample_0000_pred.png`)

## Summary

✅ **KeyError: 'image'** - FIXED (flexible loading)  
✅ **KeyError: 'mask'** - FIXED (flexible loading)  
✅ **KeyError: 'text_prompt'** - FIXED (fallback to 'prompt')  
✅ **Memory efficiency** - IMPROVED (on-demand loading)  
✅ **Prediction naming** - IMPROVED (uses sample IDs)  

The evaluation pipeline now handles both path-based and pre-loaded datasets seamlessly!
