# Klasifikasi Sampah dengan RA-CNN

**Studio Proyek 2 - Deep Learning (IF25-40401)**
Teknik Informatika, Institut Teknologi Sumatera (ITERA)

## Kelompok 7

| No | Nama Lengkap | NIM |
|----|--------------|-----|
| 1 | Devina Kartika | 123140036 |
| 2 | Muhammad Rafiq Ridho | 123140197 |
| 3 | Andryano Shevchenko Limbong | 123140205 |

Repositori: https://github.com/cengkooo/studio-2-DL-RA-CNN-klasifikasi-sampah

---

## Deskripsi Proyek

Proyek ini membangun model Artificial Intelligence berbasis Convolutional Neural Network (CNN) untuk mengklasifikasikan satu foto sampah ke dalam lima kelas yang paling sering ditemui di Indonesia. Model dievaluasi dengan metrik **Macro-F1**.

### Kelas

| ID | Kelas | Keterangan |
|----|-------|------------|
| 0 | `botol_plastik` | Botol PET bekas, punya leher botol atau tutup ulir |
| 1 | `gelas_plastik` | Gelas air mineral bersegel, cup es teh / boba, tanpa leher botol |
| 2 | `kertas_kardus` | Kertas bekas, kardus, kertas bungkus nasi, gelas kertas |
| 3 | `saset_kemasan` | Bungkus mi instan, saset kopi/sampo, bungkus makanan ringan |
| 4 | `styrofoam` | Kotak makanan styrofoam, gelas styrofoam, gabus pengemas |

Urutan kelas alfabetis, sama dengan `ImageFolder` di torchvision.

### Dataset

- Train set: sekitar 2.900 gambar (5 kelas), hasil deduplikasi, penyaringan otomatis, dan pemeriksaan manual.
- Validation set: sekitar 740 gambar, disimpan oleh tim pengajar dan tidak pernah kami lihat.
- Label masih bisa mengandung sedikit noise, sehingga penanganan noise bagian dari tantangan.
- Dataset tidak diunggah ulang ke repositori ini. Sumber gambar tercatat di `train_sources.csv`.

---

## Pendekatan Model

### Mengapa RA-CNN

Pasangan kelas seperti `botol_plastik` vs `gelas_plastik` dan `kertas_kardus` vs `styrofoam` sering hanya berbeda pada area kecil (leher botol, tutup ulir, tekstur permukaan). RA-CNN memperbaiki prediksi secara bertahap dengan memperbesar area yang paling diskriminatif.

### Arsitektur (rencana)

1. **Backbone pretrained ImageNet** (torchvision / timm) sebagai ekstraktor fitur di setiap skala.
2. **Multi-scale classification**: tiap skala punya classifier sendiri yang menghasilkan logits `[batch, 5]`.
3. **Attention Proposal Network (APN)**: memprediksi area atensi (crop) dari feature map, lalu area itu di-crop dan di-resize menjadi input skala berikutnya.
4. **Loss**:
   - Intra-scale: cross-entropy pada tiap skala.
   - Inter-scale: pairwise ranking loss agar skala yang lebih detail memberi probabilitas kelas benar yang lebih tinggi.
5. **Inferensi**: gabungan logits dari skala yang dipakai, keluaran akhir berupa logits `[batch, 5]` sesuai kontrak submission.

### Strategi Pelatihan (rencana)

- Fine-tuning bertahap: backbone dibekukan dulu, lalu dibuka dengan learning rate lebih kecil.
- Augmentasi: random resized crop, flip, rotasi ringan, color jitter.
- Penanganan imbalance kelas (kelas `kertas_kardus` paling sedikit): class weight atau weighted sampling.
- Penanganan label noise: label smoothing dan pemeriksaan sampel dengan loss tinggi.
- Validasi lokal: split `my_val/` dipotong dari train set, model dipilih berdasarkan Macro-F1 lokal.
- Pembanding: baseline ResNet-18 dari starter kit, lalu varian RA-CNN.

### Target

Baseline ResNet-18 pretrained (8 epoch) mencapai Macro-F1 sekitar **0,92** pada validation set tersembunyi. Target kami adalah melampaui angka tersebut.

---

## Hasil

> Bagian ini diisi seiring progres. Skor preview hanya umpan balik dan tidak memengaruhi nilai.

| Model | Macro-F1 (lokal) | Accuracy (lokal) | Macro-F1 (preview) |
|-------|------------------|------------------|--------------------|
| ResNet-18 baseline | - | - | - |
| RA-CNN (backbone: -) | - | - | - |

F1 per kelas dan confusion matrix akan dilampirkan setelah evaluasi lokal.

---

## Struktur Repositori

```
studio-2-DL-RA-CNN-klasifikasi-sampah/
├── README.md              # dokumen ini
├── submission/
│   ├── model.py           # get_model() dan get_transform()
│   ├── model.pth          # state_dict model terbaik
│   └── requirements.txt   # paket tambahan, mis. timm
├── evaluate.py            # salinan skrip evaluasi resmi
└── ...                    # notebook / kode pelatihan

---

## Cara Reproduksi

```bash
# 1. Clone repo
git clone https://github.com/cengkooo/studio-2-DL-RA-CNN-klasifikasi-sampah.git
cd studio-2-DL-RA-CNN-klasifikasi-sampah

# 2. Pasang dependensi
pip install torch torchvision timm pillow numpy
pip install -r submission/requirements.txt

# 3. Letakkan dataset dari Google Drive kelas ke folder train/ (jangan di-commit)

# 4. Latih model (sesuaikan nama skrip dengan yang ada di repo)
python train.py

# 5. Evaluasi pada split validasi lokal
python evaluate.py --submission ./submission --data ./my_val
```

Batas teknis yang kami patuhi: `model.pth` maksimal 500 MB, evaluasi seluruh validation set selesai dalam 15 menit. Bila bobot lebih dari 100 MB, bobot dilampirkan lewat GitHub Release atau Git LFS dan tautannya ditulis di bagian ini.

---

## Sumber Data Tambahan

Belum ada. Jika kami memakai data dari luar, sumbernya dicatat di sini.

---

## Jadwal Evaluasi

| Putaran | Tenggat commit | Fungsi |
|---------|----------------|--------|
| Preview 1 | Rabu, 14 Oktober 2026, 23:59 WIB | Umpan balik, tidak dinilai |
| Preview 2 | Rabu, 21 Oktober 2026, 23:59 WIB | Umpan balik, tidak dinilai |
| Final | Minggu, 25 Oktober 2026, 23:59 WIB | Menentukan nilai Studio 2 |

Asisten mengevaluasi commit terakhir di branch utama sebelum tenggat.

---

## Kepatuhan Aturan

- Repo publik sejak pendaftaran hingga nilai diumumkan.
- Progres dilacak lewat commit berkala.
- Dataset tidak diunggah ulang ke repositori publik, Kaggle, atau Hugging Face. Hanya kode dan bobot model.
- Tidak ada berbagi kode, bobot, atau prediksi antarkelompok.

---

## Lisensi dan Etika

Gambar dikumpulkan dari internet untuk keperluan pendidikan non-komersial di kelas ini. Repositori ini hanya berisi kode dan bobot model.