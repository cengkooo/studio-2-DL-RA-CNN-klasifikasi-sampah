"""Submission model: EfficientNetV2-S dengan head 5 kelas.

Aturan kontrak:
  - get_model() hanya MEMBANGUN arsitektur (weights=None, jangan unduh bobot di sini,
    jangan memuat model.pth di sini — evaluate.py yang memuatnya).
  - get_transform() harus SAMA dengan transform validasi saat melatih model.
  - Output model: logits [batch, 5], urutan kelas:
      0 botol_plastik, 1 gelas_plastik, 2 kertas_kardus, 3 saset_kemasan, 4 styrofoam
    (urutan alfabetis — sama dengan torchvision.datasets.ImageFolder).
"""
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5


def get_model():
    """Bangun arsitektur EfficientNetV2-S tanpa unduh bobot (sesuai kontrak evaluate.py)."""
    model = models.efficientnet_v2_s(weights=None)
    # model.classifier adalah Sequential(Dropout, Linear)
    # Ganti Linear classifier terakhir (1280 -> NUM_CLASSES)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def get_transform():
    """Transform validasi: PIL.Image (RGB) -> Tensor [3, 224, 224]."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])