import streamlit as st, pandas as pd
import modelling

df_preprocessed = (pd.read_csv("dataset-5-preprocessed.csv"))
df = (df_preprocessed[["nama_toko", "unit", "nama", "average_harga_produk_per_unit", "kategori"]].drop_duplicates(ignore_index=True))

st.write("# System Recommendation Produk di Semua Kategori Berdasarkan Harga Menggunakan Collaborative Filtering")

kategori = list(df["kategori"].unique())
pilihan_kategori = st.selectbox("Masukkan Pilihan Kategori: ", kategori)
supermarket = list(df["nama_toko"].unique())
pilihan_supermarket = st.selectbox("Masukkan Pilihan Supermarket: ", supermarket)

tombol = st.button("Lanjut")
if tombol:
    already_rated, predictions = modelling.main(df, pilihan_kategori, pilihan_supermarket)
    st.write(predictions)
