import os
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.environ.get('MODEL_PATH', 'best_vit_model.pth')
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE),
                      interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
model.head = nn.Linear(model.head.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

LABELS = ['Real', 'AI-Generated']


def predict(image):
    img = Image.fromarray(image).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)[0].cpu().numpy()
    label = LABELS[int(np.argmax(prob))]
    conf = float(np.max(prob))
    return {LABELS[0]: float(prob[0]), LABELS[1]: float(prob[1])}, \
           f"{label} ({conf * 100:.2f}%)"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='numpy'),
    outputs=[gr.Label(num_top_classes=2), gr.Textbox(label='Hasil')],
    title='Deteksi Wajah AI-Generated vs Real',
    description='Unggah foto wajah. Model ViT-Base/16 akan menebak apakah gambar dibuat AI atau asli.',
)

if __name__ == '__main__':
    demo.launch()
