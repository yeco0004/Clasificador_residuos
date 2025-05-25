import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# Rutas relativas seguras
BASE_DIR = os.path.dirname(__file__)
MODELO_PATH = os.path.join(BASE_DIR, "modelo", "modelo.h5")
CLASES_PATH = os.path.join(BASE_DIR, "modelo", "clases.json")

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model(MODELO_PATH)

# Cargar clases
@st.cache_data
def cargar_clases():
    with open(CLASES_PATH, "r") as f:
        return json.load(f)

# Ejecutar carga de recursos antes de usar
try:
    modelo = cargar_modelo()
    CLASES = cargar_clases()
except Exception as e:
    st.error(f"❌ Error cargando el modelo o las clases: {e}")
    st.stop()

# Interfaz de usuario
st.title("♻️ Clasificador de Residuos")
st.markdown("Sube una imagen de un residuo y el modelo lo clasificará.")

archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if archivo:
    try:
        imagen = Image.open(archivo).convert("RGB")
        imagen = imagen.resize((150, 150))
        arr = np.array(imagen) / 255.0
        arr = np.expand_dims(arr, axis=0)

        pred = modelo.predict(arr)
        indice = np.argmax(pred)
        confianza = np.max(pred) * 100

        st.success(f"✅ Predicción: {CLASES[indice]}")
        st.info(f"🔍 Confianza: {confianza:.2f}%")
    except Exception as e:
        st.error(f"❌ Error procesando la imagen: {e}")

