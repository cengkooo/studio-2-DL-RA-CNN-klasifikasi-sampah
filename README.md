# Studio 2 — Klasifikasi Sampah · Kelompok 7 DRA

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1 | Devina Kartika | 123140036 | Pengembangan dan pelatihan model |
| 2 | Muhammad Rafiq Ridho | 123140197 | Pengembangan model dan evaluasi |
| 3 | Andryano Shevchenko Limbong | 123140205 | Pengolahan data dan eksperimen |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: EfficientNetV2
- **Strategi pelatihan**: Pelatihan model Deep Learning menggunakan EfficientNetV2 untuk melakukan klasifikasi citra sampah.
- **Data tambahan**: Tidak ada
- **Ukuran input & preprocessing**: Citra diproses menggunakan transformasi yang disesuaikan dengan kebutuhan model EfficientNetV2.

### Kelas yang Diklasifikasikan

Model melakukan klasifikasi terhadap lima kelas sampah:

- `botol_plastik`
- `gelas_plastik`
- `kertas_kardus`
- `saset_kemasan`
- `styrofoam`

## Hasil

### Evaluasi Model Terbaik

Berikut merupakan hasil evaluasi model terbaik pada dataset validasi:

| Metrik | Hasil |
|--------|-------|
| Loss Terakhir | 0.4290 |
| Akurasi Validasi | 0.9780 (97.80%) |
| Validation Macro F1-Score | 0.9768 (97.68%) |

### F1-Score Per Kelas

| Kelas | F1-Score |
|-------|----------|
| `botol_plastik` | 0.9940 |
| `gelas_plastik` | 0.9680 |
| `kertas_kardus` | 0.9620 |
| `saset_kemasan` | 0.9800 |
| `styrofoam` | 0.9800 |

### Hasil Putaran

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | b4a940b | 0.9528 | 0.9493 |
| Preview 2 (21 Okt) | | 0.9768 | |
| Final (25 Okt) | | | |

### Catatan Eksperimen

Eksperimen dilakukan menggunakan model **EfficientNetV2-S** untuk melakukan klasifikasi citra sampah ke dalam lima kelas.

#### Parameter dan Konfigurasi Pelatihan

| Parameter / Komponen | Nilai / Konfigurasi | Keterangan |
|----------------------|---------------------|------------|
| **Arsitektur Model** | EfficientNetV2-S | Pretrained ImageNet |
| **Ukuran Citra (Input)** | 384 × 384 piksel | Resize 384 → CenterCrop 384 |
| **Batch Size** | 32 | Batch size pelatihan dan validasi |
| **Jumlah Epoch** | 30 | Model terbaik didapatkan pada Epoch 22 |
| **Optimizer** | AdamW | `weight_decay = 0.01` |
| **Learning Rate (LR)** | `2e-4` (0.0002) | Learning rate awal |
| **LR Scheduler** | CosineAnnealingLR | `T_max = 30`, `eta_min = 1e-6` |
| **Loss Function** | CrossEntropyLoss | `label_smoothing = 0.1` + Class Weighting dinamis |
| **Data Augmentation** | RandomResizedCrop (scale 0.4–1.0), Horizontal Flip (p=0.5), Rotation (15°), TrivialAugmentWide, RandomErasing (p=0.25) | Menangani citra tumpukan sampah, iklan, dan ilustrasi |
| **Data Split** | 85% Train / 15% Validation | Stratified random split |
| **Random Seed** | 42 | Menjamin hasil dapat direproduksi |

Pada iterasi **Preview 2**, dilakukan beberapa optimasi strategis:
1. **Class Weighting Dinamis:** Diterapkan untuk mengatasi disparitas data minoritas pada kelas `kertas_kardus` (311 sampel train vs 570+ sampel pada kelas lain). Hal ini berhasil mendongkrak F1 `kertas_kardus` secara drastis dari **0.9159** ke **0.9620 (+4.6%)**.
2. **Kenaikan Resolusi ke 384×384:** Menggunakan resolusi native EfficientNetV2-S untuk mempertahankan detail garis transparan `gelas_plastik` dan tekstur halus sampah, sehingga F1 `gelas_plastik` naik dari **0.9383** ke **0.9680 (+3.0%)**.
3. **Augmentasi Khusus Tumpukan & Iklan:** Integrasi `TrivialAugmentWide` dan `RandomErasing` secara dinamis mencegah model mengalami overfitting terhadap watermark, teks promosi, serta objek yang saling menutupi.
4. **Pelatihan 30 Epoch dengan Cosine Annealing:** Memberikan waktu konvergensi yang cukup bagi model untuk mempelajari augmentasi baru, mencapai performa puncak di **Epoch 22** dengan **Loss 0.4290**, **Akurasi Validasi 97.80%**, dan **Validation Macro F1-Score 97.68%**. Seluruh kelas berhasil meraih skor F1 di atas **96%**.

## Cara Reproduksi

### Instalasi Dependensi

Instal seluruh dependensi yang diperlukan menggunakan perintah berikut:
```bash
pip install -r requirements.txt
python train.py
python evaluate.py --submission ./submission --data ./my_val --out metrics.json
```

## Struktur Repositori
```bash
submission/
├── model.py          # get_model() dan get_transform()
├── model.pth         # state_dict model terbaik
└── requirements.txt  # paket/dependensi yang diperlukan

evaluate.py           # salinan skrip evaluasi resmi
train.py              # kode pelatihan model
README.md             # dokumentasi proyek
...                   # kode/notebook pelatihan kelompok
```
