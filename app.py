import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Cargar modelo y clases
modelo = tf.keras.models.load_model('keras_model.h5')

with open('clases.json', 'r') as f:
    CLASES = json.load(f)

# Título
st.title("♻️ Clasificador de Residuos")

# Cargar imagen
archivo = st.file_uploader("Sube una imagen del residuo", type=["jpg", "jpeg", "png"])

# Predicción
if archivo is not None:
    imagen = Image.open(archivo).convert('RGB')
    imagen_resized = imagen.resize((150, 150))
    arr = np.array(imagen_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = modelo.predict(arr)
    indice = np.argmax(pred)
    confianza = np.max(pred) * 100

    # Mostrar predicción en caja de texto
    st.success(f"✅ Predicción: {CLASES[indice]}")
    st.info(f"🔍 Confianza: {confianza:.2f}%")

