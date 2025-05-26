import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Cachear el modelo para cargarlo solo una vez
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/trained_model.h5")

model = load_model()

# Títulos y descripción
st.title("Clasificador de Residuos ♻️")
st.write("Sube una imagen de un residuo para clasificarlo")

# Widget para subir imágenes
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png"])

if uploaded_file is not None:
    # Preprocesamiento
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))  # Ajustar al tamaño que requiera el modelo
    img_array = np.array(image) / 255.0  # Normalización
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])
    classes = ["Orgánico", "Plástico", "Vidrio", "Papel"]  # Ajustar según las clases del modelo

    # Mostrar resultados
    st.image(image, caption="Imagen subida", use_column_width=True)
    st.success(f"Predicción: {classes[class_idx]} (Confianza: {prediction[0][class_idx]:.2f})")
