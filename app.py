import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="Clasificador de Residuos", layout="centered")

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("modelo.h5")

# Cargar clases
@st.cache_data
def cargar_clases():
    with open("clases.json", "r") as file:
        return json.load(file)

modelo = cargar_modelo()
CLASES = cargar_clases()

st.title("♻️ Clasificador de Residuos")
st.write("Sube una imagen y el modelo te dirá de qué tipo de residuo se trata.")

archivo = st.file_uploader("Selecciona una imagen...", type=["jpg", "jpeg", "png"])

if archivo:
    try:
        imagen = Image.open(archivo).convert("RGB")
        st.image(imagen, caption="Imagen subida", use_column_width=True)

        imagen = imagen.resize((150, 150))
        entrada = np.expand_dims(np.array(imagen) / 255.0, axis=0)

        predicciones = modelo.predict(entrada)
        indice = np.argmax(predicciones)
        confianza = np.max(predicciones) * 100

        st.success(f"🧠 Predicción: **{CLASES[indice]}**")
        st.info(f"Confianza: **{confianza:.2f}%**")

    except Exception as e:
        st.error(f"❌ Error procesando la imagen: {e}")
