import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import io
import numpy as np

uploaded_file = st.file_uploader("Загрузите фотографию", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение', use_column_width=True)

    if st.button("Определить вид клетки"):
        image.save("ggg.jpg")

resize = T.Resize((224, 224))
model = resnet18()
model.fc = nn.Linear(512, 4)
model.load_state_dict(torch.load('savemodel.pt', map_location=torch.device('cpu')))
model.eval()
device = 'cpu'

labels = {0: 'EOSINOPHIL', 1: 'LYMPHOCYTE', 2: 'MONOCYTE', 3: 'NEUTROPHIL'}

img = resize(io.read_image('ggg.jpg')/255)
pred = model(img.unsqueeze(0).to(device))
st.write(labels[np.argmax(pred.detach().cpu().numpy())])