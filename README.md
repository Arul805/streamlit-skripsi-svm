---
title: ABSA Ulasan Layanan Kesehatan
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.39.0"
app_file: app.py
pinned: false
---

# ABSA Ulasan Layanan Kesehatan (ML-OUSCA + SVM)

Aplikasi ini merupakan implementasi skripsi:
**"Penerapan Metode ML-OUSCA dalam Penanganan Ketidakseimbangan Data
pada Aspect-Based Sentiment Analysis (ABSA) Ulasan Layanan Kesehatan
Menggunakan SVM"**

## Fitur
- 🏠 Dashboard hasil eksperimen (perbandingan skenario, pengaruh ML-OUSCA)
- 📝 Prediksi aspek & sentimen untuk satu ulasan
- 📂 Prediksi batch untuk file .xlsx/.csv beserta visualisasi hasil

## Model
Model yang digunakan adalah SVM kernel linear dengan fitur TF-IDF (500 fitur)
pada skenario Top 5 label powerset, yang menghasilkan performa terbaik
secara keseluruhan (akurasi 81.45%, F1-Macro 52.45%).
