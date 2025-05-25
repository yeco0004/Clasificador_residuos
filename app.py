import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("keras_model.h5")

@st.cache_data
def cargar_clases():
    with open("clases.json", "r") as f:
        return json.load(f)

modelo = cargar_modelo()
CLASES = cargar_clases()

st.title("♻️ Clasificador de Residuos")

archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if archivo is not None:
    try:
        imagen = Image.open(archivo).convert("RGB")
        imagen = imagen.resize((150, 150))
        arr = np.array(imagen) / 255.0
        arr = np.expand_dims(arr, axis=0)

        pred = modelo.predict(arr)
        indice = np.argmax(pred)
        confianza = np.max(pred) * 100

        st.success(f"✅ Predicción: {CLASES[str(indice)]}")
        st.info(f"🔍 Confianza: {confianza:.2f}%")
    except Exception as e:
        st.error(f"❌ Error procesando la imagen: {e}")


