# Klasifikasi Sampah - RA-CNN & EfficientNetV2

Repositori ini berisi implementasi Klasifikasi Sampah menggunakan model Deep Learning.

## Hasil Evaluasi Model Terbaik
Berikut adalah ringkasan hasil pelatihan dan evaluasi model pada dataset validasi:

- **Loss Terakhir**: 0.4035
- **Akurasi Validasi**: 0.9561 (95.61%)
- **Validation Macro F1-Score**: 0.9528 (95.28%)

### F1-Score Per Kelas
- **botol_plastik**: 0.9804
- **gelas_plastik**: 0.9383
- **kertas_kardus**: 0.9159
- **saset_kemasan**: 0.9700
- **styrofoam**: 0.9596

## Cara Penggunaan
Jalankan evaluasi menggunakan perintah:
```bash
python evaluate.py --submission ./submission --data ./my_val --out metrics.json
```
