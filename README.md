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
| Loss Terakhir | 0.4035 |
| Akurasi Validasi | 0.9561 (95.61%) |
| Validation Macro F1-Score | 0.9528 (95.28%) |

### F1-Score Per Kelas

| Kelas | F1-Score |
|-------|----------|
| `botol_plastik` | 0.9804 |
| `gelas_plastik` | 0.9383 |
| `kertas_kardus` | 0.9159 |
| `saset_kemasan` | 0.9700 |
| `styrofoam` | 0.9596 |

### Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | b4a940b | 0.9528 | 0.9493 |
| Preview 2 (21 Okt) | | | |
| Final (25 Okt) | | | |

### Catatan Eksperimen

Eksperimen dilakukan menggunakan model **EfficientNetV2-S** untuk melakukan klasifikasi citra sampah ke dalam lima kelas.

#### Parameter dan Konfigurasi Pelatihan

| Parameter / Komponen | Nilai / Konfigurasi | Keterangan |
|----------------------|---------------------|------------|
| **Arsitektur Model** | EfficientNetV2-S | Pretrained ImageNet |
| **Ukuran Citra (Input)** | 224 × 224 piksel | Resize 256 → CenterCrop 224 |
| **Batch Size** | 32 | Batch size pelatihan dan validasi |
| **Jumlah Epoch** | 15 | Total iterasi epoch |
| **Optimizer** | AdamW | `weight_decay = 0.01` |
| **Learning Rate (LR)** | `2e-4` (0.0002) | Learning rate awal |
| **LR Scheduler** | CosineAnnealingLR | `T_max = 15`, `eta_min = 1e-6` |
| **Loss Function** | CrossEntropyLoss | `label_smoothing = 0.1` + Class Weighting dinamis |
| **Data Augmentation** | RandomResizedCrop (scale 0.7–1.0), Horizontal Flip (p=0.5), Rotation (15°), ColorJitter (0.2) | Untuk generalisasi data training |
| **Data Split** | 85% Train / 15% Validation | Stratified random split |
| **Random Seed** | 42 | Menjamin hasil dapat direproduksi |

Berdasarkan hasil evaluasi pada dataset validasi, model terbaik memperoleh **Loss sebesar 0.4035**, **akurasi validasi sebesar 95.61%**, dan **Validation Macro F1-Score sebesar 95.28%**.

Berdasarkan hasil evaluasi per kelas, performa terbaik diperoleh pada kelas `botol_plastik` dengan F1-Score sebesar **0.9804**, sedangkan F1-Score terendah terdapat pada kelas `kertas_kardus` sebesar **0.9159**.

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
