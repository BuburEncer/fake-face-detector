import os
import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from torchvision import transforms

REPO_ID = "BuburEncer/vit-fake-face-detector"
MODELS = {
    "EfficientNet-B0 (ringan)": ("best_efficientnet_b0.pth", "efficientnet_b0"),
    "ViT-Base/16 (akurat)": ("best_vit_model.pth", "vit_base_patch16_224"),
    "ResNet-50 (sedang)": ("best_resnet50.pth", "resnet50"),
}
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE),
                      interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def build_model(name):
    if name == "resnet50":
        from torchvision import models
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m
    m = timm.create_model(name, pretrained=False, num_classes=2)
    if hasattr(m, "head"):
        m.head = nn.Linear(m.head.in_features, 2)
    elif hasattr(m, "classifier"):
        m.classifier = nn.Linear(m.classifier.in_features, 2)
    return m


@st.cache_resource(show_spinner="Memuat model...")
def load_model(model_key):
    filename, arch = MODELS[model_key]
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=REPO_ID, filename=filename)
    model = build_model(arch)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


st.set_page_config(page_title="Deteksi Wajah AI vs Real", page_icon="🕵️")
st.title("🕵️ Deteksi Wajah AI-Generated vs Real")
st.markdown(
    "Model **ViT-Base/16** dilatih pada dataset 140K Real vs Fake Faces "
    "(15.000 subset) mencapai **99.23% akurasi**. Unggah foto wajah untuk diuji."
)

model_key = st.sidebar.selectbox("Pilih model", list(MODELS.keys()))
st.sidebar.caption(
    "- **EfficientNet-B0**: cepat, default\n"
    "- **ViT-Base/16**: akurasi tertinggi\n"
    "- **ResNet-50**: CNN baseline"
)

uploaded = st.file_uploader("Unggah foto wajah", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Gambar yang diunggah", width=300)

    model = load_model(model_key)
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, dim=1)[0].numpy()

    real_p, ai_p = float(prob[0]), float(prob[1])
    label = "AI-Generated" if ai_p > real_p else "Real"
    conf = max(real_p, ai_p)

    st.markdown(f"## Hasil: **{label}** ({conf * 100:.1f}%)")
    st.progress(conf)
    st.bar_chart({"Probabilitas": {"Real": real_p, "AI-Generated": ai_p}})
