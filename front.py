import streamlit as st
from PIL import Image
import requests

st.set_page_config(page_title="Распознавание Одежды")

st.title("Распознавание Одежды")
st.write("Загрузите изображение с одеждой или обувью, и модель попробует её определить.")


classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

uploaded_file = st.file_uploader(
    "Выберите изображение",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Загруженное изображение", use_container_width=True)

    if st.button("Определить Изображение"):
        with st.spinner("Идёт распознавание..."):
            try:
                files = {
                    "image": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    files=files
                )

                if response.status_code == 200:
                    result = response.json()

                    digit = result.get("Answer")  # это число (например 9)

                    # 👇 ВОТ ГЛАВНОЕ ИСПРАВЛЕНИЕ
                    class_name = classes[int(digit)]

                    st.success(f"Нейросеть считает, что это : {class_name}")

                else:
                    st.error(f"Ошибка сервера: {response.status_code}")
                    st.write(response.json())

            except Exception as e:
                st.error(f"Ошибка подключения к backend: {e}")