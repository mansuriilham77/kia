# Sistem Prediksi Permohonan KIA (Disdukcapil Kota Bogor)

Proyek ini membangun model prediksi jumlah permohonan Kartu Identitas Anak (KIA) per bulan untuk mendukung perencanaan operasional (petugas loket, ketersediaan blanko, jadwal sosialisasi/jemput bola).

Fitur utama:
- Pipeline data: validasi skema, fitur kalender (jumlah hari libur Indonesia/bulan), lag (1,2,3,6,12) dan rolling mean (3,6,12).
- Model: Baseline (Naive, Seasonal Naive) dan XGBoost (aktif jika data latih mencukupi).
- Evaluasi: MAPE, MAE, RMSE (holdout bulan terakhir).
- Dashboard Streamlit interaktif: unggah data, latih model, dan prediksi N bulan ke depan.

Struktur proyek:
```
.
├─ app/
│  └─ streamlit_app.py
├─ config/
│  └─ config.yaml
├─ data/
│  ├─ README.md
│  └─ sample_kia.csv
├─ src/
│  ├─ models/
│  │  ├─ baselines.py
│  │  ├─ evaluate.py
│  │  ├─ infer.py
│  │  └─ train.py
│  └─ pipeline/
│     └─ data_prep.py
├─ .gitignore
└─ requirements.txt
```

Persiapan:
1) (Opsional) Buat virtual env
```
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

2) Instal dependensi
```
pip install -r requirements.txt
```

Menjalankan dashboard:
```
streamlit run app/streamlit_app.py
```

Format data (CSV):
- periode: tanggal awal bulan (YYYY-MM atau YYYY-MM-01)
- permohonan_kia: jumlah permohonan (integer, >= 0)

Lihat contoh di data/sample_kia.csv dan penjelasan di data/README.md.

Catatan:
- Jika data historis < ~18–24 bulan, baseline “seasonal naive (t-12)” sering jadi acuan terbaik. XGBoost otomatis dipakai bila data memadai dan lebih baik saat evaluasi holdout.
- Artefak model disimpan di folder models/.
