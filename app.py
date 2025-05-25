import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="Clasificador IA", page_icon="🧠")

st.title("🧠 Clasificador de Imágenes con IA")
st.write("Sube una imagen para predecir la clase.")

@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model("modelo.h5")
    with open("clases.json", "r") as f:
        clases = json.load(f)
    return modelo, clases

modelo, CLASES = cargar_modelo()

imagen_subida = st.file_uploader("📷 Sube una imagen", type=["jpg", "jpeg", "png"])

if imagen_subida is not None:
    imagen = Image.open(imagen_subida).convert("RGB")
    imagen_resized = imagen.resize((150, 150))
    arr = np.array(imagen_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = modelo.predict(arr)
    indice = np.argmax(pred)
    confianza = np.max(pred) * 100

    st.success(f"✅ Predicción: {CLASES[indice]}")
    st.info(f"🔍 Confianza: {confianza:.2f}%")
