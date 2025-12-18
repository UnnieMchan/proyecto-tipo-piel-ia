import os
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# =========================
# CARGA DE MODELO Y CSV
# =========================

BASE_DIR = os.path.dirname(__file__) # carpeta donde está app.py

@st.cache_resource
def load_model():
    # Ruta 
    model_path = os.path.join(BASE_DIR, "best.pt")
    model = YOLO(model_path)
    return model

@st.cache_data
def load_products():
    csv_path = os.path.join(BASE_DIR, "productos.csv")
    productos = pd.read_csv(csv_path)
    return productos

model = load_model()
productos = load_products()

# Mapa de id de clase numérica -> nombre de tipo de piel

ID_TO_NAME = {
    0: "Grasa",
    1: "Mixta",
    2: "Seca",
    3: "Sensible"
}

# =========================
# FUNCIÓN DE RECOMENDACIÓN
# =========================
def filtrar_productos(clase_detectada, productos):

    if clase_detectada is None:
        return pd.DataFrame()

    def compatible(tipo):
        if pd.isna(tipo):
            return False
        return str(tipo).strip().lower() == str(clase_detectada).strip().lower()

    recs = productos[productos["tipo_piel_compatible"].apply(compatible)]
    # Elimina duplicados por product_id por si un mismo producto aparece varias veces
    recs = recs.drop_duplicates(subset=["product_id"])
    return recs

# =========================
# INICIALIZAR SESSION STATE
# =========================
if "imagen" not in st.session_state:
    st.session_state.imagen = None
if "mostrar_resultados" not in st.session_state:
    st.session_state.mostrar_resultados = False

# =========================
# INTERFAZ STREAMLIT
# =========================
st.title("Detección de tipo de piel y recomendación de productos")
st.write(
    "Este prototipo utiliza un modelo YOLO 11 para detectar el rostro, "
    "clasificar el tipo de piel (Grasa, Mixta, Seca, Sensible) "
    "y recomendar productos compatibles a partir del archivo productos.csv."
)

# Si ya hay resultados, mostrar botón para volver a intentar
if st.session_state.mostrar_resultados:
    if st.button("🔁 Intentar de nuevo"):
        st.session_state.imagen = None
        st.session_state.mostrar_resultados = False
        st.rerun()

# Solo mostrar la interfaz de captura si no hay resultados
if not st.session_state.mostrar_resultados:
    st.subheader("1. Captura o carga una imagen de tu rostro")
    opcion = st.radio(
        "Elige cómo proporcionar la imagen:",
        ("Usar cámara", "Subir imagen")
    )
    
    # Captura o carga de imagen
    if opcion == "Usar cámara":
        cam = st.camera_input("Toma una foto")
        if cam is not None:
            st.session_state.imagen = Image.open(cam)
            st.session_state.mostrar_resultados = True
            st.rerun()
    else:
        archivo = st.file_uploader("Sube una imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])
        if archivo is not None:
            st.session_state.imagen = Image.open(archivo)
            st.session_state.mostrar_resultados = True
            st.rerun()

# =========================
# MOSTRAR RESULTADOS
# =========================
if st.session_state.mostrar_resultados and st.session_state.imagen is not None:
    st.subheader("2. Resultado del análisis")

    imagen = st.session_state.imagen

    # Convertir a formato OpenCV (BGR)
    img_rgb = np.array(imagen)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Inference con YOLO
    results = model(img_bgr)

    clase_detectada = None

    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            # Tomamos solo la primera detección (rostro principal)
            box = boxes[0]
            cls_id = int(box.cls[0].item())
            clase_detectada = ID_TO_NAME.get(cls_id, None)

            # Dibujar caja y etiqueta en la imagen
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if clase_detectada:
                label = clase_detectada.upper()
            else:
                label = "DESCONOCIDO"

            cv2.putText(
                img_bgr,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

    if clase_detectada is None:
        st.warning("No se detectó ningún rostro o tipo de piel.")
    else:
        st.markdown(f"**Tipo de piel detectado:** {clase_detectada.upper()}")

        # Mostrar imagen de salida
        img_rgb_out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(img_rgb_out, caption="Rostro detectado y tipo de piel clasificado", use_container_width=True)

        # =========================
        # 3. MOSTRAR PRODUCTOS COMO 'TARJETAS'
        # =========================
        st.subheader("3. Productos recomendados")

        recs = filtrar_productos(clase_detectada, productos)

        if recs.empty:
            st.info("No se encontraron productos recomendados para este tipo de piel.")
        else:
            st.write(f"Se encontraron **{len(recs)}** productos recomendados para piel **{clase_detectada.lower()}**.")

            for _, row in recs.iterrows():
                col1, col2 = st.columns([1, 2])

                with col1:
                    if "url_imagen" in recs.columns and pd.notna(row["url_imagen"]):
                        st.image(row["url_imagen"], use_container_width=True)
                    else:
                        st.write("Sin imagen disponible")
                    
                with col2:
                    st.markdown(f"**Nombre:** {row['nombre_producto']}")
                    st.markdown(f"**Marca:** {row['marca']}")
                    st.markdown(f"**Categoría:** {row['categoria']}")
                    st.markdown(f"**Tipo de piel compatible:** {row['tipo_piel_compatible']}")
                    st.markdown(f"**Textura / tipo de producto:** {row['textura_tipo_producto']}")
                    st.markdown(f"**Frecuencia de uso:** {row['frecuencia_uso']}")
                    st.markdown(f"**Beneficio principal:** {row['beneficio']}")
                    st.markdown(f"**Ingredientes clave:** {row['ingredientes']}")

                st.markdown("---")

    # Botón para volver a intentar al final
    st.write("---")
    if st.button("🔁 Intentar de nuevo", key="boton_final"):
        st.session_state.imagen = None
        st.session_state.mostrar_resultados = False
        st.rerun()


# ====== EJECUTAR CON =====
# streamlit run app.py
