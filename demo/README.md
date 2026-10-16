# Deploy Demo ke Hugging Face Spaces

Demo: klasifikasi foto wajah Real vs AI-Generated dengan ViT-Base/16.

## 1. Download weights model ✅ (sudah selesai)

`best_vit_model.pth` (327MB) sudah di folder ini, didownload via Kaggle CLI:

```bash
kaggle kernels output ibnuqayyim/real-or-fake-faces-with-vit -p demo
```

```
demo/
├── app.py
├── requirements.txt
├── best_vit_model.pth       327M
└── (file output lain: *.png, *.csv, model baseline)
```

## 2. Test lokal ✅ (sudah diverifikasi)

```bash
cd demo
pip install -r requirements.txt
python app.py
```

Hasil verifikasi lokal:
- Weight load OK (2s)
- Forward ViT-Base OK
- `predict()` return format benar
- Gradio launch HTTP 200, endpoint `/gradio_api/call/predict` responsif
- Inference ~0.5s/gambar CPU

## 3. Upload weights ke Hugging Face

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload <username>/<nama-model> best_vit_model.pth
```

Ganti `<nama-model>` misal `vit-fake-face-detector`.

## 4. Buat Space

1. Buka https://huggingface.co/new-space
2. Nama: `fake-face-detector`
3. SDK: **Gradio**
4. Space hardware: Free CPU (cukup untuk 1 gambar)
5. Buat, lalu unggah `app.py` + `requirements.txt`

## 5. (Alternatif) Deploy via git

```bash
git init
git add app.py requirements.txt
git commit -m "init"
git remote add origin https://huggingface.co/spaces/<username>/<nama-space>
git push -u origin main
```

## 6. Konfigurasi MODEL_PATH (jika weights di HF hub)

Set environment variable di Settings Space:

```
MODEL_PATH = fake-face-detector/best_vit_model.pth
```

Atau kalau weights langsung di repo Space, biarkan default `best_vit_model.pth`.

## 7. Selesai

Space public → URL `https://huggingface.co/spaces/<username>/<nama-space>`.
Tempel URL itu di website portfolio.
