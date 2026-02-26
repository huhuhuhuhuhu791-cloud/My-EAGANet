"""
evaluate.py  –  EAGANet Evaluation
====================================
Cách chạy:
    python evaluate.py --checkpoint checkpoints/best_eaganet.pth
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, recall_score, matthews_corrcoef,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

sys.path.insert(0, os.path.dirname(__file__))
from dataset import EAGANetDataset
from models.Model import MambaModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_eaganet.pth")
    p.add_argument("--data_dir",   default="data")
    p.add_argument("--mask_dir",   default="")
    p.add_argument("--num_classes",type=int, default=2)
    p.add_argument("--feature_dim",type=int, default=32)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers",type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = MambaModel(in_channels=3, feature_dim=args.feature_dim)
    model.fc = nn.Linear(args.feature_dim, args.num_classes)

    assert os.path.exists(args.checkpoint), \
        f"Không tìm thấy checkpoint: {args.checkpoint}"
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Dataset
    test_dir  = os.path.join(args.data_dir, "test")
    mask_dir  = os.path.join(args.mask_dir, "test") if args.mask_dir else None
    test_set  = EAGANetDataset(test_dir, mask_dir)
    loader    = DataLoader(test_set, batch_size=args.batch_size,
                           num_workers=args.num_workers)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            prob = torch.softmax(out, dim=1).cpu().numpy()
            pred = out.argmax(dim=1).cpu().numpy()
            all_probs.extend(prob.tolist())
            all_preds.extend(pred.tolist())
            all_labels.extend(y.numpy().tolist())

    # Metrics
    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)

    print(f"\n{'='*50}")
    print(f"  Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy :  {acc:.4f}   (paper: 0.9791)")
    print(f"  MCC      :  {mcc:.4f}   (paper: 0.9487)")

    if args.num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
        print(f"  Sensitivity: {sensitivity:.4f}  (paper: 0.9731)")
        print(f"  Specificity: {specificity:.4f}  (paper: 0.9852)")
        print(f"  AUC      :  {auc:.4f}   (paper: 0.9770)")

    print(f"{'='*50}")
    print(f"\nConfusion Matrix:\n{cm}")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=test_set.classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("EAGANet Confusion Matrix")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/confusion_matrix.png", dpi=150)
    print("Saved: results/confusion_matrix.png")


if __name__ == "__main__":
    main()