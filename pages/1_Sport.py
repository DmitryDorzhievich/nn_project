import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import io
import numpy as np

uploaded_file = st.file_uploader("Загрузите фотографию", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение', use_column_width=True)

    if st.button("Определить вид спорта"):
        image.save("ggg.jpg")
        st.success("Изображение успешно сохранено на диск.")

resize = T.Resize((224, 224))
model = resnet50()
model.fc = nn.Linear(2048, 100)
model.load_state_dict(torch.load('sports.pt', map_location=torch.device('cpu')))
model.eval()
device = 'cpu'

sportname = {0: 'air hockey', 1: 'ampute football', 2: 'archery', 3: 'arm wrestling', 4: 'axe throwing', 5: 'balance beam',
 6: 'barell racing', 7: 'baseball', 8: 'basketball', 9: 'baton twirling', 10: 'bike polo', 11: 'billiards', 12: 'bmx', 13: 'bobsled',
 14: 'bowling', 15: 'boxing', 16: 'bull riding', 17: 'bungee jumping', 18: 'canoe slamon', 19: 'cheerleading', 20: 'chuckwagon racing',
 21: 'cricket', 22: 'croquet', 23: 'curling', 24: 'disc golf', 25: 'fencing', 26: 'field hockey', 27: 'figure skating men', 28: 'figure skating pairs',
 29: 'figure skating women', 30: 'fly fishing', 31: 'football', 32: 'formula 1 racing', 33: 'frisbee', 34: 'gaga', 35: 'giant slalom', 36: 'golf',
 37: 'hammer throw', 38: 'hang gliding', 39: 'harness racing', 40: 'high jump', 41: 'hockey', 42: 'horse jumping', 43: 'horse racing', 44: 'horseshoe pitching',
 45: 'hurdles', 46: 'hydroplane racing', 47: 'ice climbing', 48: 'ice yachting', 49: 'jai alai', 50: 'javelin', 51: 'jousting', 52: 'judo', 53: 'lacrosse', 54: 'log rolling',
 55: 'luge', 56: 'motorcycle racing', 57: 'mushing', 58: 'nascar racing', 59: 'olympic wrestling', 60: 'parallel bar', 61: 'pole climbing', 62: 'pole dancing',
 63: 'pole vault', 64: 'polo', 65: 'pommel horse', 66: 'rings', 67: 'rock climbing', 68: 'roller derby', 69: 'rollerblade racing', 70: 'rowing',
 71: 'rugby', 72: 'sailboat racing', 73: 'shot put', 74: 'shuffleboard', 75: 'sidecar racing', 76: 'ski jumping', 77: 'sky surfing', 78: 'skydiving',
 79: 'snow boarding', 80: 'snowmobile racing', 81: 'speed skating', 82: 'steer wrestling', 83: 'sumo wrestling', 84: 'surfing', 85: 'swimming', 86: 'table tennis', 87: 'tennis',
 88: 'track bicycle', 89: 'trapeze', 90: 'tug of war', 91: 'ultimate', 92: 'uneven bars', 93: 'volleyball', 94: 'water cycling', 95: 'water polo', 96: 'weightlifting',
 97: 'wheelchair basketball', 98: 'wheelchair racing', 99: 'wingsuit flying'}

img = resize(io.read_image('ggg.jpg')/255)
st.write(sportname[np.argmax(model(img.unsqueeze(0).to(device)).detach().cpu().numpy())])


