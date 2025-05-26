import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# Carga modelo y clases al inicio, fuera de interacción
modelo = tf.keras.models.load_model('modelo.h5')
with open('clases.json', 'r') as f:
    CLASES = json.load(f)

st.title("Clasificador de Residuos")

archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

resultado_placeholder = st.empty()

if archivo:
    with st.spinner('Procesando la imagen...'):
        try:
            imagen = Image.open(archivo).convert("RGB")
            imagen_resized = imagen.resize((150, 150))
            arr = np.array(imagen_resized) / 255.0
            arr = np.expand_dims(arr, axis=0)

            pred = modelo.predict(arr)
            indice = np.argmax(pred)
            confianza = np.max(pred) * 100

            with resultado_placeholder.container():
                st.image(imagen, caption="Imagen subida", use_column_width=True)
                if indice < len(CLASES):
                    st.success(f"✅ Predicción: **{CLASES[indice]}**")
                    st.info(f"🔍 Confianza: **{confianza:.2f}%**")
                else:
                    st.warning("⚠️ Índice de predicción fuera de rango.")

        except Exception as e:
            resultado_placeholder.error(f"Error procesando la imagen: {e}")
else:
    resultado_placeholder.empty()

