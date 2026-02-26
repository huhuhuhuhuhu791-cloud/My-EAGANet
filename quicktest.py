"""
quick_test.py  –  Kiểm tra model chạy được không (không cần dataset)
======================================================================
Chạy:  python quick_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

def test_model():
    print("="*50)
    print("  EAGANet Quick Test (Không cần dataset)")
    print("="*50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Device: {device}")

    # Import model
    from models.Model import MambaModel
    print("✓ Model import OK")

    # Khởi tạo model
    num_classes = 2   # benign / malignant
    feature_dim = 32
    model = MambaModel(in_channels=3, feature_dim=feature_dim, num_lwm_mamba_blocks=3)

    # ✅ Sửa output layer
    model.fc = nn.Linear(feature_dim, num_classes)
    model = model.to(device)
    model.eval()
    print(f"✓ Model khởi tạo OK  (output={num_classes} classes)")

    # Đếm params
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"✓ Total params:     {total_params:.2f}M")
    print(f"✓ Trainable params: {trainable:.2f}M")

    # Test forward pass với 4 channels (RGB + density mask)
    print("\n--- Test Forward Pass ---")
    batch_size = 2

    # Input 4 channels
    x4 = torch.randn(batch_size, 4, 224, 224).to(device)
    with torch.no_grad():
        out = model(x4)
    print(f"✓ Input  [4ch]: {x4.shape} → Output: {out.shape}")
    assert out.shape == (batch_size, num_classes), "Output shape sai!"

    # Input 3 channels (model tự sinh density)
    x3 = torch.randn(batch_size, 3, 224, 224).to(device)
    with torch.no_grad():
        out3 = model(x3)
    print(f"✓ Input  [3ch]: {x3.shape} → Output: {out3.shape}")

    # Test FGSM
    print("\n--- Test FGSM Attack ---")
    model.train()
    x_adv = x4.clone().requires_grad_(True)
    y = torch.zeros(batch_size, dtype=torch.long).to(device)
    loss = nn.CrossEntropyLoss()(model(x_adv), y)
    loss.backward()
    x_adv_new = x_adv + 0.03 * x_adv.grad.sign()
    print(f"✓ FGSM OK: {x_adv_new.shape}")

    # Test FLOPS (nếu có thop)
    try:
        from thop import profile
        model.eval()
        flops, params = profile(model, inputs=(x4[:1],), verbose=False)
        print(f"\n✓ FLOPs:  {flops/1e9:.2f} GFLOPS  (paper: 6.5G)")
        print(f"✓ Params: {params/1e6:.2f}M  (paper: 32M)")
    except ImportError:
        print("\n⚠ thop chưa cài (pip install thop) — bỏ qua FLOPS")

    print("\n" + "="*50)
    print("  ✅ Tất cả tests PASS! Model sẵn sàng train.")
    print("="*50)
    print("\nBước tiếp theo:")
    print("  1. Chuẩn bị data:  data/train/benign/  data/train/malignant/")
    print("                     data/test/benign/   data/test/malignant/")
    print("  2. Chạy train:     python train.py")
    print("  3. Đánh giá:       python evaluate.py --checkpoint checkpoints/best_eaganet.pth")


if __name__ == "__main__":
    test_model()