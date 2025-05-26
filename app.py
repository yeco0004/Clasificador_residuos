import streamlit as st

def main():
    if "show_button" not in st.session_state:
        st.session_state.show_button = True

    st.checkbox("Mostrar botón", key="show_button")

    # Siempre renderizamos ambos widgets (o al menos mantenemos el espacio)
    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.show_button:
            if st.button("Botón visible"):
                st.write("Botón visible presionado")
        else:
            st.write("Botón oculto")

    with col2:
        st.write("Otro contenido estable")

if __name__ == "__main__":
    main()
