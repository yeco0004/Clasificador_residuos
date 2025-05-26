import streamlit as st
from PIL import Image

st.title("Prueba básica")

uploaded_file = st.file_uploader("Sube imagen", type=["png", "jpg"])

placeholder = st.empty()

if uploaded_file:
    img = Image.open(uploaded_file)
    placeholder.image(img)
else:
    placeholder.empty()

