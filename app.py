import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

st.set_page_config(page_title="Clasificador de Residuos")

@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("modelo.h5")

@st.cache_data
def cargar_clases():
    with open("clases.json", "r") as f:
        return json.load(f)

try:
    modelo = cargar_modelo()
    CLASES = cargar_clases()
except Exception as e:
    st.error(f"Error cargando el modelo o las clases: {e}")
    st.stop()

st.title("♻️ Clasificador de Residuos")

archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

# Crear placeholders vacíos para imagen y resultados
imagen_placeholder = st.empty()
resultado_placeholder = st.empty()

if archivo:
    try:
        imagen = Image.open(archivo).convert("RGB")
        imagen_placeholder.image(imagen, caption="Imagen subida", use_column_width=True)

        imagen_resized = imagen.resize((150, 150))
        arr = np.array(imagen_resized) / 255.0
        arr = np.expand_dims(arr, axis=0)

        pred = modelo.predict(arr)
        indice = np.argmax(pred)
        confianza = np.max(pred) * 100

        if indice < len(CLASES):
            resultado_placeholder.success(f"✅ Predicción: **{CLASES[indice]}**")
            resultado_placeholder.info(f"🔍 Confianza: **{confianza:.2f}%**")
        else:
            resultado_placeholder.warning("⚠️ Índice de predicción fuera de rango.")

    except Exception as e:
        resultado_placeholder.error(f"Error procesando la imagen: {e}")
else:
    # Si no hay archivo, limpia los placeholders
    imagen_placeholder.empty()
    resultado_placeholder.empty()
