import streamlit as st
from PIL import Image, ImageOps

# Title
st.title("Hello, Streamlit!")

# Header
st.header("Welcome to my first Streamlit app")

# Text
st.write("This is a simple Streamlit app example.")

if 'switch' not in st.session_state:
    st.session_state.switch = 0

image_placeholder = st.empty()
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
   image_org = Image.open(uploaded_file)
   image_placeholder.image(image_org, caption='Brand:',use_column_width=True)