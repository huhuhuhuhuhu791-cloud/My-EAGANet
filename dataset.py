"""
dataset.py  –  EAGANet Dataset
==============================
Cấu trúc thư mục mong đợi:

  data/
  ├── train/
  │   ├── benign/          ← ảnh PNG/JPG
  │   └── malignant/
  ├── train_mask/          ← density mask tương ứng (cùng tên file)
  │   ├── benign/
  │   └── malignant/
  ├── test/
  │   ├── benign/
  │   └── malignant/
  └── test_mask/
      ├── benign/
      └── malignant/

Nếu KHÔNG có mask thư mục, GradCAM sẽ tự sinh channel 4.
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class EAGANetDataset(Dataset):
    """
    Trả về tensor [4, 224, 224]:
      - channels 0-2: RGB mammogram (normalize ImageNet)
      - channel 3   : breast density mask (0-1) hoặc zeros nếu không có
    """
    def __init__(self, image_dir: str, mask_dir: str = None,
                 image_size: tuple = (224, 224), use_gradcam: bool = False):
        self.image_size = image_size
        self.mask_dir = mask_dir
        self.use_gradcam = use_gradcam
        self.samples = []  # (image_path, mask_path_or_None, label)

        # Đọc classes từ sub-folder
        self.classes = sorted([
            d for d in os.listdir(image_dir)
            if os.path.isdir(os.path.join(image_dir, d))
        ])
        assert len(self.classes) > 0, f"Không tìm thấy class folder trong {image_dir}"
        print(f"[Dataset] Classes: {self.classes}")

        for label, cls in enumerate(self.classes):
            cls_img_dir = os.path.join(image_dir, cls)
            for fname in sorted(os.listdir(cls_img_dir)):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img_path = os.path.join(cls_img_dir, fname)

                # Tìm mask tương ứng
                mask_path = None
                if mask_dir:
                    candidate = os.path.join(mask_dir, cls, fname)
                    if os.path.exists(candidate):
                        mask_path = candidate

                self.samples.append((img_path, mask_path, label))

        print(f"[Dataset] Tổng {len(self.samples)} ảnh từ '{image_dir}'")

        # Transform cho ảnh
        self.img_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # GradCAM preprocessor (lazy load)
        self._gradcam_preprocessor = None

    def _get_gradcam_preprocessor(self):
        if self._gradcam_preprocessor is None:
            # Import ở đây để tránh circular import
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from common.Process import GradCAMPreprocessor
            self._gradcam_preprocessor = GradCAMPreprocessor(self.image_size)
        return self._gradcam_preprocessor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]

        # --- Load ảnh ---
        pil_img = Image.open(img_path).convert('RGB')

        if self.use_gradcam and mask_path is None:
            # Dùng GradCAM sinh channel 4
            tensor_4ch = self._get_gradcam_preprocessor()(pil_img)  # [4,H,W]
        else:
            # Channel 0-2: RGB normalize
            img_tensor = self.img_transform(pil_img)  # [3, H, W]

            # Channel 3: density mask (0~1 float)
            if mask_path and os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, self.image_size,
                                  interpolation=cv2.INTER_NEAREST)
                mask_tensor = torch.from_numpy(mask / 255.0).float().unsqueeze(0)
            else:
                mask_tensor = torch.zeros(1, *self.image_size)

            tensor_4ch = torch.cat([img_tensor, mask_tensor], dim=0)  # [4, H, W]

        return tensor_4ch, torch.tensor(label, dtype=torch.long)