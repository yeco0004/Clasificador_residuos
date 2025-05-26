# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="♻️ Clasificador de Residuos", layout="centered")

# ===== Cargar modelo y clases =====
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
    st.error(f"❌ Error cargando el modelo o las clases: {e}")
    st.stop()

# ===== Interfaz =====
st.title("♻️ Clasificador de Residuos")
st.write("Sube una imagen de un residuo para que el modelo lo clasifique.")

archivo = st.file_uploader("Selecciona una imagen...", type=["jpg", "jpeg", "png"])

if archivo is not None:
    imagen = Image.open(archivo).convert("RGB")
    st.image(imagen, caption="🖼️ Imagen cargada", use_column_width=True)

    if st.button("🔍 Clasificar"):
        try:
            # Preprocesar imagen
            imagen_redimensionada = imagen.resize((150, 150))
            entrada = np.expand_dims(np.array(imagen_redimensionada) / 255.0, axis=0)

            # Realizar predicción
            pred = modelo.predict(entrada)
            indice = int(np.argmax(pred))
            confianza = float(np.max(pred)) * 100

            # Mostrar resultados
            if indice < len(CLASES):
                st.success(f"✅ Predicción: **{CLASES[indice]}**")
                st.info(f"🔬 Confianza: **{confianza:.2f}%**")
            else:
                st.warning("⚠️ Índice fuera del rango de clases.")

        except Exception as e:
            st.error(f"❌ Error durante la predicción: {e}")

            st.error("❌ Error procesando la imagen.")
            st.exception(e)

