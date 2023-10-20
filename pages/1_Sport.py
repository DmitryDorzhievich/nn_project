import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import io

uploaded_file = st.file_uploader("Загрузите фотографию", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение', use_column_width=True)

    if st.button("Сохранить на диск"):
        image.save("ggg.jpg")
        st.success("Изображение успешно сохранено на диск.")

resize = T.Resize((224, 224))
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.fc = nn.Linear(2048, 100)
model.eval()
device = 'cpu'

img = resize(io.read_image('ggg.jpg')/255)
st.write(model(img.unsqueeze(0).to(device)))


