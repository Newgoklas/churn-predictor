# 📡 Churn Predictor — UAS Data Science
### Sales & Marketing Dataset | Machine Learning Pipeline + Modern Dashboard

---

## 📋 Tentang Proyek

Aplikasi prediksi **Customer Churn** lengkap dari pipeline data science (EDA → preprocessing → modeling → tuning) hingga dashboard Streamlit modern bertema dark "ops console" dengan gauge chart interaktif.

> **Catatan penting:** Versi ini memperbaiki masalah pada file sebelumnya di mana `top_features.pkl` (5 fitur) tidak sinkron dengan `scaler.pkl`/`best_model.pkl` (10 fitur), yang menyebabkan error saat prediksi. Sekarang **main.py** memastikan seluruh artefak (`model`, `scaler`, `top_features`) selalu memiliki jumlah fitur yang sama persis.

---

## 🗂️ Struktur File

```
uas_final/
├── main.py                # Pipeline lengkap: EDA → Modeling → Tuning → Save Model
├── app.py                 # Dashboard Streamlit modern (3 tab)
├── requirements.txt
├── README.md
├── sales_marketing.csv    # Dataset (dibuat otomatis jika belum ada)
│
├── output_plots/          # Visualisasi EDA & evaluasi
│   ├── 1_missing_values.png
│   ├── 2_distribusi_churn.png
│   ├── 3_heatmap_korelasi.png
│   ├── 4_feature_importance.png
│   └── 5_confusion_matrix_best.png
│
└── models/                 # Artefak model (SELALU konsisten satu sama lain)
    ├── best_model.pkl
    ├── scaler.pkl           # fit pada top_features
    ├── scaler_top.pkl       # alias dari scaler.pkl
    ├── label_encoders.pkl
    ├── top_features.pkl     # fitur yang dipakai model & scaler
    ├── all_features.pkl
    └── model_metadata.pkl
```

---

## ⚙️ Cara Menjalankan

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Jalankan pipeline (buat dataset jika belum ada, latih & simpan model)
python main.py

# 3. Jalankan dashboard
streamlit run app.py
```

Buka browser ke **http://localhost:8501**

> Jika `sales_marketing.csv` tidak ada, dataset sintetis 15.000 baris dibuat otomatis dengan struktur yang sama seperti dataset asli Kaggle.

---

## 🧠 Isi Pipeline (`main.py`)

| Tahap | Deskripsi |
|---|---|
| 1. EDA | head, info, describe, missing value, distribusi churn, heatmap korelasi |
| 2. Direct Modeling | Logistic Regression, Random Forest, Voting Classifier — tanpa preprocessing |
| 3. Preprocessing | drop kolom tidak relevan, hapus duplikat, imputasi median/modus, outlier IQR, label encoding, scaling setelah split |
| 4. Feature Selection | feature importance dari Random Forest → ambil top-10 fitur |
| 5. Hyperparameter Tuning | GridSearchCV (cv=3–5, scoring=F1) untuk ketiga model pada fitur top-10 |
| 6. Simpan Model | model + scaler + fitur + metadata disimpan **dengan jumlah fitur yang sama** |

---

## 🖥️ Dashboard (`app.py`)

### Desain
Tema **dark "ops console"** — navy/slate background, aksen indigo/ungu, kartu kaca (glassmorphism ringan), tipografi DM Sans + DM Mono. Elemen ciri khas: **gauge chart risiko churn** sebagai focal point hasil prediksi.

### Fitur
- **Tab Prediksi Tunggal** — form otomatis menyesuaikan fitur yang dipakai model, tombol isi cepat (Acak / Contoh Churn / Contoh Loyal), indikator risiko cepat sebelum prediksi, hasil dengan gauge chart + rekomendasi tindakan kontekstual.
- **Tab Prediksi Batch** — upload CSV, prediksi massal, ringkasan grafik (pie + histogram), download hasil.
- **Tab Riwayat & Analisis** — log semua prediksi dalam sesi, statistik agregat, grafik distribusi.
- **Validasi otomatis** — app akan menampilkan pesan jelas jika artefak model tidak ditemukan atau tidak sinkron, alih-alih error mentah.

### Cara mengisi form (mudah dipahami)
Setiap input memiliki ikon, label dalam Bahasa Indonesia, dan keterangan singkat (hint) di bawahnya — misalnya "1 = sangat tidak puas · 10 = sangat puas" — sehingga pengguna awam pun langsung paham cara mengisinya tanpa instruksi tambahan.

---

## 📈 Hasil Model (contoh, akan bervariasi tiap run)

| Model | Accuracy | F1-Score |
|---|---|---|
| Logistic Regression (Tuned) | ~0.80 | ~0.86 |
| Random Forest (Tuned) | ~0.80 | ~0.86 |
| Voting Classifier (Tuned) | ~0.80 | ~0.86 |

Model dengan F1-Score tertinggi otomatis dipilih dan disimpan sebagai `best_model.pkl`.

---

## 🔧 Troubleshooting

| Masalah | Solusi |
|---|---|
| `Model belum dibuat` di app | Jalankan `python main.py` terlebih dahulu |
| `Artefak tidak sinkron` | Hapus folder `models/` lalu jalankan ulang `python main.py` |
| Error saat upload CSV batch | Pastikan kolom CSV sama persis dengan fitur yang ditampilkan di app (lihat sidebar "Fitur Aktif") |
