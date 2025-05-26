import streamlit as st
from PIL import Image

st.title("Prueba básica de subida y muestra de imagen")

uploaded_file = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

placeholder = st.empty()

if uploaded_file:
    img = Image.open(uploaded_file)
    placeholder.image(img, caption="Imagen subida", use_column_width=True)
else:
    placeholder.empty()

