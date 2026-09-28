import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
import matplotlib.pyplot as plt

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================

st.set_page_config(
    page_title="ABSA Ulasan Layanan Kesehatan",
    page_icon="🏥",
    layout="wide"
)

# =====================================================
# LOAD MODEL & FILE PENDUKUNG (CACHED SUPAYA TIDAK RELOAD TERUS)
# =====================================================

@st.cache_resource
def load_model():
    with open("model/2500_top5_svm.pkl", "rb") as f:
        model = pickle.load(f)
    return model


@st.cache_resource
def load_vectorizer():
    with open("model/2500_top5_tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer


@st.cache_resource
def load_label_mapping():
    with open("model/label_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
    return mapping


@st.cache_resource
def load_slang_dict():
    slang_df = pd.read_csv("model/colloquial-indonesian-lexicon.csv")
    slang_dict = dict(zip(
        slang_df['slang'].astype(str),
        slang_df['formal'].astype(str)
    ))
    return slang_dict


@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()


@st.cache_resource
def load_stopwords():
    kata_penting = [
        'tidak', 'bukan', 'belum', 'tanpa', 'tetapi', 'tapi', 'namun',
        'hanya', 'melainkan', 'sedang', 'sedangkan', 'padahal',
        'sebaliknya', 'meski', 'meskipun', 'walau', 'walaupun',
        'kendati', 'biar', 'biarpun', 'sekalipun', 'sungguhpun'
    ]
    stop_factory = StopWordRemoverFactory()
    stopwords = set(stop_factory.get_stop_words())
    stopwords = stopwords - set(kata_penting)
    return stopwords


model = load_model()
vectorizer = load_vectorizer()
label_mapping = load_label_mapping()
slang_dict = load_slang_dict()
stemmer = load_stemmer()
stopwords = load_stopwords()


# =====================================================
# PIPELINE PREPROCESSING (SAMA SEPERTI SAAT TRAINING)
# =====================================================

def case_folding(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'@\w+|#\w+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenizing(text):
    return text.split()


def normalizing(tokens):
    return [slang_dict.get(word, word) for word in tokens]


def stemming(tokens):
    return [stemmer.stem(word) for word in tokens]


def filtering(tokens):
    return [word for word in tokens if word not in stopwords]


def preprocess_text(text):
    text = case_folding(text)
    tokens = tokenizing(text)
    tokens = normalizing(tokens)
    tokens = stemming(tokens)
    tokens = filtering(tokens)
    return ' '.join(tokens)


def predict_text(raw_text):
    """
    Mengambil teks mentah -> preprocessing -> TF-IDF -> prediksi SVM
    -> decode ke nama aspek & sentimen.
    Mengembalikan dict berisi teks bersih, kode label, dan deskripsinya.
    """
    clean = preprocess_text(raw_text)
    X = vectorizer.transform([clean]).toarray()
    pred_code = model.predict(X)[0]
    deskripsi = label_mapping.get(pred_code, "Label tidak dikenali")

    return {
        "clean_text": clean,
        "label_code": pred_code,
        "deskripsi": deskripsi
    }


# =====================================================
# SIDEBAR -- NAVIGASI MENU
# =====================================================

st.sidebar.title("🏥 ABSA Layanan Kesehatan")
menu = st.sidebar.radio(
    "Pilih Menu",
    ["🏠 Dashboard", "📝 Prediksi Satu Ulasan", "📂 Prediksi Dataset (.xlsx/.csv)"]
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Skripsi: Penerapan Metode ML-OUSCA dalam Penanganan "
    "Ketidakseimbangan Data pada Aspect-Based Sentiment Analysis "
    "(ABSA) Ulasan Layanan Kesehatan Menggunakan SVM"
)

# =====================================================
# MENU 1 -- DASHBOARD
# =====================================================

if menu == "🏠 Dashboard":

    st.title("🏠 Dashboard Hasil Penelitian")
    st.markdown(
        "Ringkasan hasil eksperimen klasifikasi sentimen berbasis aspek "
        "pada ulasan layanan kesehatan menggunakan **SVM + Label Powerset**, "
        "dengan dan tanpa penerapan **ML-OUSCA**."
    )

    st.subheader("Model yang Digunakan untuk Prediksi Live")
    col1, col2, col3 = st.columns(3)
    col1.metric("Skenario", "Top 5 Label")
    col2.metric("Akurasi (Test)", "81.45%")
    col3.metric("F1-Macro (Test)", "52.45%")

    st.markdown("---")

    st.subheader("Perbandingan Seluruh Skenario Pengujian")

    ringkasan = pd.DataFrame({
        "Skenario": [
            "Top 5 (tanpa ML-OUSCA)", "Top 5 (ML-OUSCA)",
            "Top 10 (tanpa ML-OUSCA)", "Top 10 (ML-OUSCA)",
            "Top 20 (tanpa ML-OUSCA)", "Top 20 (ML-OUSCA)",
            "Top 30 (tanpa ML-OUSCA)", "Top 30 (ML-OUSCA)",
        ],
        "Akurasi (%)": [81.45, 81.45, 74.55, 75.23, 67.72, 67.93, 68.38, 68.38],
        "Precision-Macro (%)": [50.73, 50.73, 39.33, 41.35, 14.14, 14.16, 12.65, 13.55],
        "Recall-Macro (%)": [54.71, 54.71, 30.47, 33.74, 13.76, 13.79, 10.96, 11.58],
        "F1-Macro (%)": [52.45, 52.45, 30.80, 34.56, 12.93, 12.95, 10.12, 11.05],
    })

    st.dataframe(ringkasan, use_container_width=True)
    st.caption(
        "Tabel di atas merupakan ringkasan performa pada data uji (test set) "
        "untuk seluruh skenario pengujian yang dilakukan."
    )

    st.markdown("---")

    st.subheader("Pengaruh ML-OUSCA terhadap F1-Macro")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    label_variasi = ["Top 5", "Top 10", "Top 20", "Top 30"]
    tanpa = [52.45, 30.80, 12.93, 10.12]
    dengan = [52.45, 34.56, 12.95, 11.05]

    x = np.arange(len(label_variasi))
    width = 0.35

    ax.bar(x - width / 2, tanpa, width, label="Tanpa ML-OUSCA")
    ax.bar(x + width / 2, dengan, width, label="Dengan ML-OUSCA")
    ax.set_xticks(x)
    ax.set_xticklabels(label_variasi)
    ax.set_ylabel("F1-Macro (%)")
    ax.set_title("Perbandingan F1-Macro Sebelum dan Sesudah ML-OUSCA")
    ax.legend()

    st.pyplot(fig)

    st.info(
        "Model yang dipakai untuk fitur prediksi pada aplikasi ini adalah "
        "skenario **Top 5 tanpa ML-OUSCA**, karena menghasilkan performa "
        "terbaik secara keseluruhan dan ML-OUSCA tidak mengubah distribusi "
        "data pada variasi label ini."
    )

    st.markdown("---")

    st.subheader("Daftar Label yang Dapat Dikenali Sistem")
    st.markdown(
        "Karena model dilatih menggunakan **Top 5 label powerset terbanyak**, "
        "sistem ini hanya dapat memprediksi salah satu dari 5 kombinasi "
        "aspek-sentimen berikut. Ulasan dengan kombinasi aspek lain di luar "
        "daftar ini akan tetap diklasifikasikan ke label yang paling mendekati."
    )

    label_top5_info = pd.DataFrame({
        "Kode Label": ["c2", "c3", "c5", "c6", "c23"],
        "Kombinasi Aspek & Sentimen": [
            label_mapping.get("c2", "-"),
            label_mapping.get("c3", "-"),
            label_mapping.get("c5", "-"),
            label_mapping.get("c6", "-"),
            label_mapping.get("c23", "-"),
        ],
        "Jumlah Data Uji": [18, 167, 177, 21, 16],
    })

    st.dataframe(label_top5_info, use_container_width=True, hide_index=True)

    st.caption(
        "Kolom 'Jumlah Data Uji' menunjukkan jumlah sampel masing-masing label "
        "pada data pengujian (test set), yang mencerminkan tingkat "
        "ketidakseimbangan data pada skenario Top 5."
    )


# =====================================================
# MENU 2 -- PREDIKSI SATU ULASAN
# =====================================================

elif menu == "📝 Prediksi Satu Ulasan":

    st.title("📝 Prediksi Satu Ulasan")
    st.markdown(
        "Masukkan teks ulasan layanan kesehatan, sistem akan memprediksi "
        "kombinasi aspek dan sentimen yang terkandung di dalamnya."
    )

    user_text = st.text_area(
        "Tulis ulasan di sini:",
        height=150,
        placeholder="Contoh: Pelayanan dokternya ramah dan fasilitas rumah sakit bersih, tapi waktu tunggunya lama sekali..."
    )

    if st.button("🔍 Prediksi", type="primary"):
        if user_text.strip() == "":
            st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
        else:
            with st.spinner("Memproses..."):
                hasil = predict_text(user_text)

            st.markdown("---")
            st.subheader("Hasil Prediksi")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Teks Asli**")
                st.write(user_text)
            with col2:
                st.markdown("**Teks Setelah Preprocessing**")
                st.write(hasil["clean_text"] if hasil["clean_text"] else "_(kosong setelah preprocessing)_")

            st.markdown("**Aspek & Sentimen Terdeteksi**")
            st.success(hasil["deskripsi"])

            with st.expander("Detail teknis"):
                st.write(f"Kode label powerset: `{hasil['label_code']}`")


# =====================================================
# MENU 3 -- PREDIKSI DATASET
# =====================================================

elif menu == "📂 Prediksi Dataset (.xlsx/.csv)":

    st.title("📂 Prediksi Dataset")
    st.markdown(
        "Upload file berisi banyak ulasan sekaligus (.xlsx atau .csv) "
        "untuk diprediksi secara batch."
    )

    uploaded_file = st.file_uploader(
        "Upload file di sini",
        type=["xlsx", "csv"]
    )

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):
            df_upload = pd.read_csv(uploaded_file)
        else:
            df_upload = pd.read_excel(uploaded_file)

        st.subheader("Preview Data")
        st.dataframe(df_upload.head(), use_container_width=True)

        kolom_teks = st.selectbox(
            "Pilih kolom yang berisi teks ulasan:",
            df_upload.columns
        )

        if st.button("🚀 Proses Prediksi Batch", type="primary"):

            with st.spinner(f"Memproses {len(df_upload)} baris data..."):

                hasil_list = []
                progress_bar = st.progress(0)

                for i, teks in enumerate(df_upload[kolom_teks].astype(str)):
                    hasil = predict_text(teks)
                    hasil_list.append(hasil)
                    progress_bar.progress((i + 1) / len(df_upload))

                df_hasil = df_upload.copy()
                df_hasil["clean_text"] = [h["clean_text"] for h in hasil_list]
                df_hasil["label_code"] = [h["label_code"] for h in hasil_list]
                df_hasil["aspek_sentimen"] = [h["deskripsi"] for h in hasil_list]

            st.success(f"Selesai memprediksi {len(df_upload)} ulasan.")

            st.markdown("---")
            st.subheader("Hasil Prediksi")
            st.dataframe(df_hasil, use_container_width=True)

            # =================================================
            # VISUALISASI HASIL PREDIKSI DATASET
            # =================================================

            st.markdown("---")
            st.subheader("Visualisasi Distribusi Hasil Prediksi")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Distribusi Kode Label Powerset**")
                label_counts = df_hasil["label_code"].value_counts()

                fig1, ax1 = plt.subplots(figsize=(5, 4))
                ax1.bar(label_counts.index.astype(str), label_counts.values)
                ax1.set_xlabel("Kode Label")
                ax1.set_ylabel("Jumlah Ulasan")
                ax1.set_title("Distribusi Label Powerset Hasil Prediksi")
                plt.xticks(rotation=45)
                st.pyplot(fig1)

            with col2:
                st.markdown("**Proporsi Tiap Kombinasi Aspek-Sentimen**")
                aspek_counts = df_hasil["aspek_sentimen"].value_counts()

                fig2, ax2 = plt.subplots(figsize=(5, 4))
                ax2.pie(
                    aspek_counts.values,
                    labels=aspek_counts.index,
                    autopct="%1.1f%%",
                    textprops={"fontsize": 7}
                )
                ax2.set_title("Proporsi Aspek-Sentimen")
                st.pyplot(fig2)

            # =================================================
            # DOWNLOAD HASIL
            # =================================================

            st.markdown("---")
            csv_hasil = df_hasil.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Hasil Prediksi (.csv)",
                data=csv_hasil,
                file_name="hasil_prediksi_absa.csv",
                mime="text/csv"
            )