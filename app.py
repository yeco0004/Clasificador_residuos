import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# --------- Carga del modelo y clases con manejo seguro de errores ---------
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("modelo/keras_modelo.h5")

@st.cache_data
def cargar_clases():
    with open("modelo/clases.json", "r") as f:
        return json.load(f)

st.set_page_config(page_title="Clasificador de Residuos", layout="centered")

st.title("♻️ Clasificador de Residuos")
st.markdown("Sube una imagen de un residuo y el modelo lo clasificará.")

# ---------- Inicialización segura del modelo y clases ----------
modelo = None
CLASES = None

with st.spinner("Cargando modelo..."):
    try:
        modelo = cargar_modelo()
        CLASES = cargar_clases()
    except Exception as e:
        st.error("❌ Error cargando el modelo o las clases. Verifica que existan los archivos correctamente.")
        st.exception(e)
        st.stop()

# ----------- Subida de imagen y predicción -------------------
archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if archivo is not None:
    with st.spinner("Procesando imagen..."):
        try:
            imagen = Image.open(archivo).convert("RGB")
            imagen = imagen.resize((150, 150))
            arr = np.array(imagen) / 255.0
            arr = np.expand_dims(arr, axis=0)

            pred = modelo.predict(arr)
            indice = int(np.argmax(pred))
            confianza = float(np.max(pred)) * 100

            st.image(imagen, caption="Imagen cargada", use_column_width=True)
            st.success(f"✅ Predicción: {CLASES[indice]}")
            st.info(f"🔍 Confianza del modelo: {confianza:.2f}%")

        except Exception as e:
            st.error("❌ Error procesando la imagen.")
            st.exception(e)

