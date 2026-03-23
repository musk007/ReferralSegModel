"""
Evaluation script for trained DualProtoSeg model.
Usage: python evaluate.py --checkpoint <path_to_checkpoint> --config <config_file> --split <test|valid>
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from src.model import ClsNetwork
from src.conch_adapter import ConchAdapter
from utils.hierarchical_utils import merge_to_parent_predictions
from utils.pyutils import set_seed
from utils.trainutils import get_cls_dataset
from utils.validate import validate_test, generate_cam
from conch.open_clip_custom import create_model_from_pretrained
from torch.cuda.amp import autocast


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained DualProtoSeg model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "valid", "val"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation (default: from config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for predictions (default: next to checkpoint)",
    )
    parser.add_argument(
        "--generate_cams",
        action="store_true",
        help="Generate and save CAM visualizations",
    )
    return parser.parse_args()


def get_device(gpu_id: int):
    return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


def build_clip_model(cfg, device):
    """Build CONCH adapter model."""
    clip_cfg = getattr(cfg, "clip", None)
    clip_cfg = OmegaConf.to_container(clip_cfg, resolve=True) if clip_cfg is not None else {}

    model_name = clip_cfg.get("model_name", "conch_ViT-B-16")
    checkpoint_path = clip_cfg.get("checkpoint_path")
    hf_hub = clip_cfg.get("hf_hub", "MahmoodLab/conch")

    # BCSS class prompts (same as training)
    bcss_conceptual_prompts = {
        0: [  # tumor
        "Invasive carcinoma composed of solid nests and cords of pleomorphic atypical cells infiltrating desmoplastic stroma without myoepithelial rims.",
        "Segment densely cellular invasive carcinoma composed of solid sheets and irregular nests infiltrating desmoplastic stroma without myoepithelial rims.",
        "Segment cohesive sheets and nests of markedly atypical epithelial cells with high nuclear grade infiltrating fibrous stroma, lacking myoepithelial rims.",
        "Segment invasive carcinoma forming solid nests and cords with marked nuclear atypia infiltrating fibrous stroma, lacking myoepithelial rims.",
        "Segment invasive carcinoma composed of solid nests and cords of pleomorphic atypical cells infiltrating desmoplastic stroma without myoepithelial rims.",
        "Segment invasive carcinoma composed of pleomorphic atypical cells in solid nests and cords infiltrating desmoplastic stroma, lacking myoepithelial rims.",
        "Segment cohesive sheets and nests of markedly atypical epithelial cells infiltrating stroma without myoepithelial rim or ductal confinement.",
        "Segment sheets and nests of highly atypical invasive carcinoma cells infiltrating stroma without myoepithelial rim or ductal confinement.",
        "Segment infiltrative nests and cords of atypical epithelial cells with angulated glands invading desmoplastic stroma, lacking myoepithelial rims.",
        "Segment invasive carcinoma formed by cohesive solid nests and cords of pleomorphic, high-grade epithelial cells infiltrating desmoplastic stroma without myoepithelial rims or ductal confinement."
    ],
    1: [  # stroma
        "Isolate fibrotic tumor-associated stroma with collagen bundles and reactive spindle fibroblasts cuffing and separating invasive carcinoma nests, with only sparse inflammatory cells.",
        "Isolate fibrotic tumor-associated stroma showing collagenized desmoplasia with scattered spindle fibroblasts adjacent to invasive carcinoma, lacking dense lymphoid aggregates.",
        "Isolate fibrotic desmoplastic stroma interdigitating between invasive tumor nests, composed of collagen bundles with scattered bland spindle fibroblasts and minimal inflammation.",
        "Segment desmoplastic tumor-associated stroma composed of collagenized fibroblastic tissue separating invasive carcinoma nests, with sparse scattered inflammatory cells.",
        "Isolate fibrotic tumor-associated stroma with dense collagen bundles and reactive spindle fibroblasts separating invasive carcinoma nests, lacking dense lymphoid aggregates.",
        "Isolate tumor-associated desmoplastic stroma showing dense collagen with reactive spindle fibroblasts cuffing and separating invasive carcinoma nests, with minimal inflammatory infiltrate.",
        "Isolate fibrotic desmoplastic stroma with spindle fibroblasts and collagen separating invasive tumor nests, containing only scattered inflammatory cells.",
        "Isolate fibrotic desmoplastic stroma with reactive spindle fibroblasts cuffing and separating invasive tumor nests, lacking dense lymphoid aggregates.",
        "Isolate fibrotic desmoplastic stroma with collagen bundles and reactive spindle cells cuffing and separating invasive tumor nests, with sparse inflammatory cells.",
        "Isolate tumor-associated desmoplastic stroma composed of dense collagen bundles and reactive spindle fibroblasts interposed between invasive carcinoma nests, with only sparse inflammatory cells."
    ],
    2: [  # inflammatory
        "Segment inflamed stroma showing dense lymphocytic infiltrates and loose edematous connective tissue without prominent collagenized desmoplasia or cohesive epithelial tumor nests.",
        "Isolate loose edematous stroma densely infiltrated by small lymphocytes and plasma cells, lacking prominent collagenized desmoplastic reaction.",
        "Segment inflamed stroma characterized by loose edematous connective tissue heavily infiltrated by lymphocytes and plasma cells, without prominent collagenized desmoplasia.",
        "Select inflamed stroma with dense lymphocytic infiltrate and loose edematous matrix adjacent to tumor, lacking prominent desmoplastic collagenization.",
        "Select stromal regions with dense lymphocytic infiltrates and loose edematous matrix, often forming periductal or peritumoral aggregates, lacking prominent collagen bundles.",
        "Identify inflamed stroma containing dense sheets of small lymphocytes and plasma cells within loose edematous connective tissue, lacking cohesive atypical epithelial nests.",
        "Identify loose edematous stroma densely infiltrated by lymphocytes and plasma cells, lacking prominent collagen bundles or tumor cell nests.",
        "Select stroma densely infiltrated by small lymphocytes and plasma cells forming sheets and aggregates within loose edematous connective tissue, without cohesive tumor nests.",
        "Segment inflamed stroma composed of loose, edematous connective tissue with dense sheets of lymphocytes and plasma cells, lacking prominent collagenized desmoplasia or epithelial tumor nests.",
        "Identify lymphocyte- and plasma cell-rich inflamed stroma with a loose edematous matrix, without dense collagen bundles or cohesive atypical epithelial clusters."
    ],
    3: [  # necrosis
        "Select central acellular necrotic zone with ghost tumor cell outlines and karyorrhectic debris not confined within pre-existing ducts.",
        "Select acellular eosinophilic necrotic debris with ghost outlines and karyorrhectic material not confined within ductal or glandular structures.",
        "Isolate central geographic necrosis showing eosinophilic ghost cells and karyorrhectic debris within a tumor nodule, not confined by ductal walls.",
        "Select acellular eosinophilic necrotic area with ghost outlines and karyorrhectic debris not confined within ducts, surrounded by viable tumor and stroma.",
        "Identify acellular necrotic debris forming irregular geographic zones within tumor, with ghost outlines and karyorrhectic material not confined to ducts.",
        "Segment acellular eosinophilic necrotic pools with ghost tumor cell outlines and karyorrhectic debris, not confined within ducts or glands.",
        "Identify acellular necrotic area with eosinophilic debris and ghost outlines not confined within ducts, surrounded by viable invasive carcinoma and reactive stroma.",
        "Segment central geographic necrosis composed of acellular eosinophilic debris with ghost cell outlines and karyorrhectic material, lacking confinement within ductal or glandular structures.",
        "Identify irregular intratumoral necrotic zones showing eosinophilic ghost outlines and fragmented nuclear debris, not restricted to pre-existing ducts.",
        "Select acellular necrotic regions with homogeneous eosinophilic debris and ghost tumor cell contours, occurring freely within tumor rather than within ductal spaces."
    ]
    }
    class_prompts = bcss_conceptual_prompts

    clip_adapter = ConchAdapter(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device=device,
        class_prompts=class_prompts,
        prompt_n_ctx=16,
        prompt_position="end",
        freeze_conch=True,
        hf_hub=hf_hub,
    )
    clip_adapter.to(device)
    return clip_adapter


def build_model(cfg, device, clip_adapter=None):
    """Build the segmentation model."""
    bcss_conceptual_prompts = {
        0: [  # tumor
        "Invasive carcinoma composed of solid nests and cords of pleomorphic atypical cells infiltrating desmoplastic stroma without myoepithelial rims.",
        "Segment densely cellular invasive carcinoma composed of solid sheets and irregular nests infiltrating desmoplastic stroma without myoepithelial rims.",
        "Segment cohesive sheets and nests of markedly atypical epithelial cells with high nuclear grade infiltrating fibrous stroma, lacking myoepithelial rims.",
        "Segment invasive carcinoma forming solid nests and cords with marked nuclear atypia infiltrating fibrous stroma, lacking myoepithelial rims.",
        "Segment invasive carcinoma composed of solid nests and cords of pleomorphic atypical cells infiltrating desmoplastic stroma without myoepithelial rims.",
        "Segment invasive carcinoma composed of pleomorphic atypical cells in solid nests and cords infiltrating desmoplastic stroma, lacking myoepithelial rims.",
        "Segment cohesive sheets and nests of markedly atypical epithelial cells infiltrating stroma without myoepithelial rim or ductal confinement.",
        "Segment sheets and nests of highly atypical invasive carcinoma cells infiltrating stroma without myoepithelial rim or ductal confinement.",
        "Segment infiltrative nests and cords of atypical epithelial cells with angulated glands invading desmoplastic stroma, lacking myoepithelial rims.",
        "Segment invasive carcinoma formed by cohesive solid nests and cords of pleomorphic, high-grade epithelial cells infiltrating desmoplastic stroma without myoepithelial rims or ductal confinement."
    ],
    1: [  # stroma
        "Isolate fibrotic tumor-associated stroma with collagen bundles and reactive spindle fibroblasts cuffing and separating invasive carcinoma nests, with only sparse inflammatory cells.",
        "Isolate fibrotic tumor-associated stroma showing collagenized desmoplasia with scattered spindle fibroblasts adjacent to invasive carcinoma, lacking dense lymphoid aggregates.",
        "Isolate fibrotic desmoplastic stroma interdigitating between invasive tumor nests, composed of collagen bundles with scattered bland spindle fibroblasts and minimal inflammation.",
        "Segment desmoplastic tumor-associated stroma composed of collagenized fibroblastic tissue separating invasive carcinoma nests, with sparse scattered inflammatory cells.",
        "Isolate fibrotic tumor-associated stroma with dense collagen bundles and reactive spindle fibroblasts separating invasive carcinoma nests, lacking dense lymphoid aggregates.",
        "Isolate tumor-associated desmoplastic stroma showing dense collagen with reactive spindle fibroblasts cuffing and separating invasive carcinoma nests, with minimal inflammatory infiltrate.",
        "Isolate fibrotic desmoplastic stroma with spindle fibroblasts and collagen separating invasive tumor nests, containing only scattered inflammatory cells.",
        "Isolate fibrotic desmoplastic stroma with reactive spindle fibroblasts cuffing and separating invasive tumor nests, lacking dense lymphoid aggregates.",
        "Isolate fibrotic desmoplastic stroma with collagen bundles and reactive spindle cells cuffing and separating invasive tumor nests, with sparse inflammatory cells.",
        "Isolate tumor-associated desmoplastic stroma composed of dense collagen bundles and reactive spindle fibroblasts interposed between invasive carcinoma nests, with only sparse inflammatory cells."
    ],
    2: [  # inflammatory
        "Segment inflamed stroma showing dense lymphocytic infiltrates and loose edematous connective tissue without prominent collagenized desmoplasia or cohesive epithelial tumor nests.",
        "Isolate loose edematous stroma densely infiltrated by small lymphocytes and plasma cells, lacking prominent collagenized desmoplastic reaction.",
        "Segment inflamed stroma characterized by loose edematous connective tissue heavily infiltrated by lymphocytes and plasma cells, without prominent collagenized desmoplasia.",
        "Select inflamed stroma with dense lymphocytic infiltrate and loose edematous matrix adjacent to tumor, lacking prominent desmoplastic collagenization.",
        "Select stromal regions with dense lymphocytic infiltrates and loose edematous matrix, often forming periductal or peritumoral aggregates, lacking prominent collagen bundles.",
        "Identify inflamed stroma containing dense sheets of small lymphocytes and plasma cells within loose edematous connective tissue, lacking cohesive atypical epithelial nests.",
        "Identify loose edematous stroma densely infiltrated by lymphocytes and plasma cells, lacking prominent collagen bundles or tumor cell nests.",
        "Select stroma densely infiltrated by small lymphocytes and plasma cells forming sheets and aggregates within loose edematous connective tissue, without cohesive tumor nests.",
        "Segment inflamed stroma composed of loose, edematous connective tissue with dense sheets of lymphocytes and plasma cells, lacking prominent collagenized desmoplasia or epithelial tumor nests.",
        "Identify lymphocyte- and plasma cell-rich inflamed stroma with a loose edematous matrix, without dense collagen bundles or cohesive atypical epithelial clusters."
    ],
    3: [  # necrosis
        "Select central acellular necrotic zone with ghost tumor cell outlines and karyorrhectic debris not confined within pre-existing ducts.",
        "Select acellular eosinophilic necrotic debris with ghost outlines and karyorrhectic material not confined within ductal or glandular structures.",
        "Isolate central geographic necrosis showing eosinophilic ghost cells and karyorrhectic debris within a tumor nodule, not confined by ductal walls.",
        "Select acellular eosinophilic necrotic area with ghost outlines and karyorrhectic debris not confined within ducts, surrounded by viable tumor and stroma.",
        "Identify acellular necrotic debris forming irregular geographic zones within tumor, with ghost outlines and karyorrhectic material not confined to ducts.",
        "Segment acellular eosinophilic necrotic pools with ghost tumor cell outlines and karyorrhectic debris, not confined within ducts or glands.",
        "Identify acellular necrotic area with eosinophilic debris and ghost outlines not confined within ducts, surrounded by viable invasive carcinoma and reactive stroma.",
        "Segment central geographic necrosis composed of acellular eosinophilic debris with ghost cell outlines and karyorrhectic material, lacking confinement within ductal or glandular structures.",
        "Identify irregular intratumoral necrotic zones showing eosinophilic ghost outlines and fragmented nuclear debris, not restricted to pre-existing ducts.",
        "Select acellular necrotic regions with homogeneous eosinophilic debris and ghost tumor cell contours, occurring freely within tumor rather than within ductal spaces."
    ]
    }
    
    model = ClsNetwork(
        backbone=cfg.model.backbone.config,
        stride=cfg.model.backbone.stride,
        cls_num_classes=cfg.dataset.cls_num_classes,
        clip_adapter=clip_adapter,
        pretrained=False,  # Not needed for evaluation
        enable_text_fusion=getattr(cfg.model, "enable_text_fusion", True),
        text_prompts=bcss_conceptual_prompts,
        fusion_dim=getattr(cfg.model, "fusion_dim", None),
        spatial_agg_all_scales=True,
    )
    return model.to(device)


def load_checkpoint(model, checkpoint_path, device):
    """Load model weights from checkpoint."""
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
        print("✓ Model weights loaded successfully")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print("✓ Model weights loaded (direct state dict)")
    
    # Print checkpoint info
    if "epoch" in checkpoint:
        print(f"  Checkpoint from epoch: {checkpoint['epoch']}")
    if "best_mIoU" in checkpoint:
        print(f"  Best mIoU during training: {checkpoint['best_mIoU']:.4f}")
    
    return model


def evaluate(args):
    """Main evaluation function."""
    print("\n" + "="*80)
    print("DUALPROTOSEG MODEL EVALUATION")
    print("="*80)
    
    # Setup
    set_seed(42)
    device = get_device(args.gpu)
    print(f"\nDevice: {device}")
    
    # Load config
    cfg = OmegaConf.load(args.config)
    print(f"Config loaded: {args.config}")
    
    # Setup output directory
    if args.output_dir is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        args.output_dir = os.path.join(checkpoint_dir, f"eval_{args.split}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Build models
    print("\nBuilding models...")
    clip_model = build_clip_model(cfg, device)
    model = build_model(cfg, device, clip_adapter=clip_model)
    
    # Load checkpoint
    model = load_checkpoint(model, args.checkpoint, device)
    model.eval()
    
    # Prepare dataset
    print(f"\nPreparing {args.split} dataset...")
    num_workers = min(10, os.cpu_count())
    batch_size = args.batch_size if args.batch_size is not None else cfg.train.samples_per_gpu
    
    # Normalize split name
    split = "test" if args.split == "test" else "valid"
    _, test_dataset = get_cls_dataset(cfg, split=split, enable_rotation=False, p=0.0)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    print(f"Dataset size: {len(test_dataset)} samples")
    print(f"Batch size: {batch_size}")
    
    # Evaluation metrics
    print("\n" + "-"*80)
    print("COMPUTING SEGMENTATION METRICS")
    print("-"*80)
    
    loss_function = nn.BCEWithLogitsLoss().to(device)
    
    # Run segmentation evaluation
    test_mIoU, test_mean_dice, test_fw_iu, test_iu_per_class, test_dice_per_class = validate_test(
        model=model,
        data_loader=test_loader,
        cfg=cfg,
        cls_loss_func=loss_function,
    )
    
    # Compute classification metrics
    print("\n" + "-"*80)
    print("COMPUTING CLASSIFICATION METRICS")
    print("-"*80)
    
    all_preds = []
    all_labels = []
    val_loss_sum = 0.0
    val_batch_count = 0
    val_correct_all = 0
    val_total_samples = 0
    
    with torch.no_grad():
        for _, inputs, cls_labels, _ in tqdm(test_loader, desc="Classification eval", ncols=100):
            inputs = inputs.to(device).float()
            cls_labels = cls_labels.to(device).float()
            
            with autocast():
                outputs = model(inputs)
                cls4 = outputs[6]
                k_list = outputs[9]
                cls4_merge = merge_to_parent_predictions(cls4, k_list, method=cfg.train.merge_test)
                
                val_cls_loss = loss_function(cls4_merge, cls_labels)
                val_loss_sum += val_cls_loss.item()
                val_batch_count += 1
                
                cls_pred4 = (torch.sigmoid(cls4_merge) > 0.5).float()
                all_correct = (cls_pred4 == cls_labels).all(dim=1).float()
                val_correct_all += all_correct.sum().item()
                val_total_samples += all_correct.shape[0]
                
                all_preds.append(torch.sigmoid(cls4_merge).cpu().numpy())
                all_labels.append(cls_labels.cpu().numpy())
    
    avg_val_loss = val_loss_sum / val_batch_count if val_batch_count > 0 else 0.0
    val_acc = (val_correct_all / val_total_samples * 100) if val_total_samples > 0 else 0.0
    
    if len(all_preds) > 0:
        val_preds = np.concatenate(all_preds, axis=0)
        val_labels = np.concatenate(all_labels, axis=0)
        try:
            val_auc = roc_auc_score(val_labels, val_preds, average="macro")
        except Exception:
            val_auc = 0.0
    else:
        val_auc = 0.0
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"\n{'CLASSIFICATION METRICS':^80}")
    print("-"*80)
    print(f"  Loss:     {avg_val_loss:.4f}")
    print(f"  Accuracy: {val_acc:.2f}%")
    print(f"  AUC:      {val_auc:.4f}")
    
    print(f"\n{'SEGMENTATION METRICS':^80}")
    print("-"*80)
    print(f"  mIoU:       {test_mIoU:.4f}")
    print(f"  Mean Dice:  {test_mean_dice:.4f}")
    print(f"  FwIU:       {test_fw_iu:.4f}")
    
    print(f"\n{'PER-CLASS METRICS':^80}")
    print("-"*80)
    class_names = ["Tumor", "Stroma", "Inflammatory", "Necrosis", "Background"]
    print(f"  {'Class':<15} {'IoU':<10} {'Dice':<10}")
    print("  " + "-"*35)
    for i, (iou, dice) in enumerate(zip(test_iu_per_class, test_dice_per_class)):
        class_name = class_names[i] if i < len(class_names) else f"Class {i}"
        print(f"  {class_name:<15} {iou:.4f}    {dice:.4f}")
    
    # Save results to file
    results_file = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(results_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("DUALPROTOSEG EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Dataset size: {len(test_dataset)}\n\n")
        
        f.write("CLASSIFICATION METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Loss:     {avg_val_loss:.4f}\n")
        f.write(f"Accuracy: {val_acc:.2f}%\n")
        f.write(f"AUC:      {val_auc:.4f}\n\n")
        
        f.write("SEGMENTATION METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"mIoU:       {test_mIoU:.4f}\n")
        f.write(f"Mean Dice:  {test_mean_dice:.4f}\n")
        f.write(f"FwIU:       {test_fw_iu:.4f}\n\n")
        
        f.write("PER-CLASS METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Class':<15} {'IoU':<10} {'Dice':<10}\n")
        f.write("-"*35 + "\n")
        for i, (iou, dice) in enumerate(zip(test_iu_per_class, test_dice_per_class)):
            class_name = class_names[i] if i < len(class_names) else f"Class {i}"
            f.write(f"{class_name:<15} {iou:.4f}    {dice:.4f}\n")
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Generate CAMs if requested
    if args.generate_cams:
        print("\n" + "-"*80)
        print("GENERATING CAM VISUALIZATIONS")
        print("-"*80)
        
        cam_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        # Temporarily set pred_dir for CAM generation
        original_pred_dir = cfg.work_dir.pred_dir if hasattr(cfg.work_dir, 'pred_dir') else None
        cfg.work_dir.pred_dir = os.path.join(args.output_dir, "cams")
        os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
        
        print(f"Generating CAMs for {len(test_dataset)} samples...")
        print(f"Output directory: {cfg.work_dir.pred_dir}")
        
        generate_cam(model=model, data_loader=cam_loader, cfg=cfg)
        
        print(f"✓ CAMs saved to: {cfg.work_dir.pred_dir}")
        
        # Restore original pred_dir
        if original_pred_dir:
            cfg.work_dir.pred_dir = original_pred_dir
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80 + "\n")


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
