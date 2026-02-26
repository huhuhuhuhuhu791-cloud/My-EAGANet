"""
train.py  –  EAGANet Training Script
=====================================
Cách chạy:
    python train.py

Hoặc tuỳ chỉnh:
    python train.py --data_dir data --epochs 200 --num_classes 2
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, recall_score, matthews_corrcoef,
    roc_auc_score, confusion_matrix
)
import numpy as np

# ── đảm bảo import đúng path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from dataset import EAGANetDataset
from models.Model import MambaModel


# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Train EAGANet")
    p.add_argument("--data_dir",    default="data",    help="Thư mục chứa train/ test/")
    p.add_argument("--mask_dir",    default="",        help="Thư mục mask (để trống nếu không có)")
    p.add_argument("--epochs",      type=int, default=200)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-5)
    p.add_argument("--feature_dim", type=int, default=32)
    p.add_argument("--num_classes", type=int, default=2,
                   help="2=benign/malignant, 5=BI-RADS 3-4A-4B-4C-5")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir",    default="checkpoints")
    p.add_argument("--use_gradcam", action="store_true",
                   help="Dùng GradCAM tự sinh channel 4 (chậm hơn)")
    # Adversarial
    p.add_argument("--alpha",  type=float, default=1.0,  help="Weight clean loss")
    p.add_argument("--beta",   type=float, default=0.5,  help="Weight adversarial loss")
    p.add_argument("--eps",    type=float, default=0.03, help="FGSM epsilon")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
def fgsm_attack(model, x, y, criterion, eps):
    """REF strategy: FGSM adversarial example (Eq.16 trong paper)"""
    x_adv = x.clone().detach().requires_grad_(True)
    loss = criterion(model(x_adv), y)
    loss.backward()
    with torch.no_grad():
        x_adv = x_adv + eps * x_adv.grad.sign()
        x_adv = x_adv.clamp(0, 1)
    return x_adv.detach()


# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(all_labels, all_preds, all_probs, num_classes):
    acc = accuracy_score(all_labels, all_preds)
    cm  = confusion_matrix(all_labels, all_preds)

    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        mcc = matthews_corrcoef(all_labels, all_preds)
        try:
            auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
        except Exception:
            auc = 0.0
        return dict(acc=acc, sensitivity=sensitivity,
                    specificity=specificity, mcc=mcc, auc=auc)
    else:
        mcc = matthews_corrcoef(all_labels, all_preds)
        return dict(acc=acc, mcc=mcc)


# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion,
                    device, epoch, total_epochs, args):
    model.train()
    total_loss = 0.0
    adversarial_phase = (epoch >= total_epochs // 2)  # 100 epoch đầu clean

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if adversarial_phase:
            # Adversarial training: L = α*CE(clean) + β*CE(adv)  — Eq.21
            out_clean = model(x)
            loss_clean = criterion(out_clean, y)

            x_adv = fgsm_attack(model, x, y, criterion, args.eps)
            out_adv = model(x_adv)
            loss_adv = criterion(out_adv, y)

            loss = args.alpha * loss_clean + args.beta * loss_adv
        else:
            out = model(x)
            loss = criterion(out, y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    phase_str = "Adversarial" if adversarial_phase else "Clean"
    avg_loss = total_loss / len(loader)
    return avg_loss, phase_str


# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for x, y in loader:
        x = x.to(device)
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(y.numpy().tolist())

    return compute_metrics(all_labels, all_preds, all_probs, num_classes)


# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  EAGANet Training")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}  |  Classes: {args.num_classes}")
    print(f"{'='*60}\n")

    # ── Paths ──────────────────────────────────────────────────────────────
    train_img_dir  = os.path.join(args.data_dir, "train")
    test_img_dir   = os.path.join(args.data_dir, "test")
    train_mask_dir = os.path.join(args.mask_dir, "train") if args.mask_dir else None
    test_mask_dir  = os.path.join(args.mask_dir, "test")  if args.mask_dir else None

    assert os.path.exists(train_img_dir), \
        f"Không tìm thấy: {train_img_dir}\n" \
        "Hãy tạo cấu trúc:  data/train/benign/  và  data/train/malignant/"
    assert os.path.exists(test_img_dir), \
        f"Không tìm thấy: {test_img_dir}"

    # ── Dataset ────────────────────────────────────────────────────────────
    train_set = EAGANetDataset(
        train_img_dir, train_mask_dir,
        image_size=(224, 224), use_gradcam=args.use_gradcam
    )
    test_set = EAGANetDataset(
        test_img_dir, test_mask_dir,
        image_size=(224, 224), use_gradcam=args.use_gradcam
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = MambaModel(
        in_channels=3,
        feature_dim=args.feature_dim,
        num_lwm_mamba_blocks=3
    )

    # ✅ FIX: sửa output layer theo num_classes (code gốc hardcode 28)
    model.fc = nn.Linear(args.feature_dim, args.num_classes)
    model = model.to(device)

    # In thông số model
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {total_params:.2f}M\n")

    # ── Optimizer & Loss ───────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Paper dùng StepLR (lr_drop=40 theo config)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=40, gamma=0.5
    )
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0
    log_lines = ["epoch,phase,loss,acc,sensitivity,specificity,mcc,auc\n"]

    # ── Training Loop ──────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        avg_loss, phase = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, args.epochs, args
        )
        scheduler.step()
        metrics = evaluate(model, test_loader, device, args.num_classes)

        acc  = metrics["acc"]
        sens = metrics.get("sensitivity", 0)
        spec = metrics.get("specificity", 0)
        mcc  = metrics.get("mcc", 0)
        auc  = metrics.get("auc", 0)

        print(
            f"Epoch [{epoch+1:03d}/{args.epochs}] "
            f"[{phase:11s}] "
            f"Loss: {avg_loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"Sen: {sens:.4f} | "
            f"Spe: {spec:.4f} | "
            f"MCC: {mcc:.4f} | "
            f"AUC: {auc:.4f}"
        )

        log_lines.append(
            f"{epoch+1},{phase},{avg_loss:.4f},"
            f"{acc:.4f},{sens:.4f},{spec:.4f},{mcc:.4f},{auc:.4f}\n"
        )

        # Lưu best model
        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(args.save_dir, "best_eaganet.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model → {save_path}  (acc={best_acc:.4f})")

        # Lưu mỗi 50 epoch
        if (epoch + 1) % 50 == 0:
            ckpt = os.path.join(args.save_dir, f"eaganet_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt)

    # ── Save log ───────────────────────────────────────────────────────────
    log_path = os.path.join(args.save_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.writelines(log_lines)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best Accuracy: {best_acc:.4f}")
    print(f"  Log saved  → {log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()