import streamlit as st
from PIL import Image
from prediction import load_trained_model,predict_fruit

st.title(":orange[FreshHarvest: AI-Based Fresh and Spoiled Fruit Classification]")

st.subheader("Upload your fruit image")

model = load_trained_model()

display_names = { "F_Banana": "Fresh Banana", "S_Banana": "Spoiled Banana", "F_Lemon": "Fresh Lemon", "S_Lemon": "Spoiled Lemon", "F_Lulo": "Fresh Lulo", "S_Lulo": "Spoiled Lulo", "F_Mango": "Fresh Mango", "S_Mango": "Spoiled Mango", "F_Orange": "Fresh Orange", "S_Orange": "Spoiled Orange", "F_Strawberry": "Fresh Strawberry", "S_Strawberry": "Spoiled Strawberry", "F_Tamarillo": "Fresh Tamarillo", "S_Tamarillo": "Spoiled Tamarillo", "F_Tomato": "Fresh Tomato", "S_Tomato": "Spoiled Tomato" }

uploaded_file = st.file_uploader(
    "Drag and drop your fruit image here",
    type=["jpg", "jpeg", "png", "webp", "bmp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.success("Image successfully uploaded!")
        st.image(image, caption="Uploaded Fruit Image", use_container_width=True)

        # Prediction
        predicted_label = predict_fruit(model,image)
        predicted_name = display_names[predicted_label]

        st.subheader("Prediction Result")
        st.success(f"{predicted_name}")

    except Exception as e:
        st.error(f"Error loading image: {e}")


