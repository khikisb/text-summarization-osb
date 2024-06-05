import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud

# Mengunduh stopwords NLTK jika belum diunduh
try:
    nltk.data.find('corpora/stopwords.zip')
except:
    nltk.download('stopwords')

# Menggunakan stopwords bahasa Indonesia
try:
    stop_words = stopwords.words('indonesian')
except:
    nltk.download('stopwords')
    stop_words = stopwords.words('indonesian')

# Fungsi untuk membaca teks artikel berdasarkan ID artikel
def baca_teks_artikel(teks_artikel):
    artikel = teks_artikel.split(". ")
    kalimat = []
    for kal in artikel:
        kalimat.append(kal.replace("[^a-zA-Z]", " ").split(" "))
    kalimat.pop()
    return kalimat

# Fungsi untuk membuat matriks kemiripan antar kalimat
def buat_matriks_kemiripan(kalimat, stop_words):
    matriks_kemiripan = np.zeros((len(kalimat), len(kalimat)))
    for indeks1 in range(len(kalimat)):
        for indeks2 in range(len(kalimat)):
            if indeks1 == indeks2:
                continue
            matriks_kemiripan[indeks1][indeks2] = kemiripan_kalimat(kalimat[indeks1], kalimat[indeks2], stop_words)
    return matriks_kemiripan

# Fungsi untuk menghitung kemiripan antar kalimat
def kemiripan_kalimat(kal1, kal2, stopwords=None):
    if stopwords is None:
        stopwords = []
    kal1 = [w.lower() for w in kal1]
    kal2 = [w.lower() for w in kal2]
    semua_kata = list(set(kal1 + kal2))
    vektor1 = [0] * len(semua_kata)
    vektor2 = [0] * len(semua_kata)
    for w in kal1:
        if w in stopwords:
            continue
        vektor1[semua_kata.index(w)] += 1
    for w in kal2:
        if w in stopwords:
            continue
        vektor2[semua_kata.index(w)] += 1
    return 1 - cosine_distance(vektor1, vektor2)

# Fungsi untuk menghasilkan word cloud dari teks ringkasan
def buat_word_cloud(teks):
    wordcloud = WordCloud(width=1600, height=800).generate(teks)
    plt.figure(figsize=(16, 8), facecolor='k')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig('wordcloud.png', facecolor='k', bbox_inches='tight')

# Fungsi untuk meringkas teks artikel
def ringkas_teks(teks_artikel, top_n=3):
    teks_ringkasan = []
    kalimat = baca_teks_artikel(teks_artikel)
    matriks_kemiripan_kalimat = buat_matriks_kemiripan(kalimat, stop_words)
    graph_kemiripan_kalimat = nx.from_numpy_array(matriks_kemiripan_kalimat)
    skor = nx.pagerank(graph_kemiripan_kalimat)
    kalimat_terurut = sorted(((skor[i], s) for i, s in enumerate(kalimat)), reverse=True)
    for i in range(top_n):
        teks_ringkasan.append(" ".join(kalimat_terurut[i][1]))
    ringkasan = ". ".join(teks_ringkasan)
    buat_word_cloud(ringkasan)
    return ringkasan, graph_kemiripan_kalimat

# Aplikasi Streamlit
st.title("Perangkum dan Visualisasi Artikel Berita")

# Memuat data dari CSV
url = "https://gist.githubusercontent.com/khikisb/ce2f0cedd1605c056966bec0396f35ad/raw/3089ba335874a6818927dab3309727c68eedde98/berita.csv"
df = pd.read_csv(url)

# Tata letak Tab
tab1, tab2 = st.tabs(["Ringkasan Semua Artikel", "Ringkasan Artikel Kustom"])

with tab1:
    st.header("Ringkasan Semua Artikel")

    # Menampilkan tabel dengan pagination
    per_page = 10
    total_pages = len(df) // per_page + (1 if len(df) % per_page > 0 else 0)
    page = st.number_input("Halaman", min_value=1, max_value=total_pages, step=1, value=1)
    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, len(df))
    st.write(df.iloc[start_idx:end_idx])

    # Tombol untuk meringkas setiap artikel
    for i in range(start_idx, end_idx):
        if st.button(f"Ringkas Artikel {i}"):
            ringkasan, _ = ringkas_teks(df.loc[i, 'isi-berita'])
            df.at[i, 'ringkasan'] = ringkasan

with tab2:
    st.header("Ringkas Artikel Kustom")
    teks_artikel = st.text_area("Masukkan teks artikel untuk diringkas")
    top_n = st.number_input("Masukkan jumlah kalimat untuk ringkasan", min_value=1, max_value=10, step=1, value=3)

    if st.button("Ringkas"):
        ringkasan, graph = ringkas_teks(teks_artikel, top_n)
        st.subheader("Ringkasan")
        st.write(ringkasan)
        
        st.subheader("Grafik Kemiripan Kalimat")
        plt.figure(figsize=(10, 7))
        nx.draw(graph, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray', font_size=20, font_weight='bold')
        st.pyplot(plt)

        st.subheader("Word Cloud")
        image = Image.open('wordcloud.png')
        st.image(image, use_column_width=True)
