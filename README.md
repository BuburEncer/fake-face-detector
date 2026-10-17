# Klasifikasi Citra AI-Generated vs Real (Vision Transformer)

Proyek UAS Computer Vision: klasifikasi citra wajah AI-generated vs citra asli menggunakan **ViT-Base/16** dengan transfer learning dari ImageNet-21k, dibandingkan dengan baseline CNN (ResNet-50, EfficientNet-B0).

## Dataset

- **Sumber:** [AI Generated Images vs Real Images (140K Real and Fake Faces Dataset)](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- **Struktur:** `real_vs_fake/train/real` dan `real_vs_fake/train/fake`
- **Subset yang dipakai:** 7500 gambar per kelas (total 15.000) agar RAM aman
- **Label:** `1` = AI-Generated, `0` = Real

## Arsitektur Model

| Parameter           | Nilai                                |
| ------------------- | ------------------------------------ |
| Model               | ViT-Base/16 (`vit_base_patch16_224`) |
| Pretrained          | ImageNet-21k                         |
| Patch size          | 16×16 (grid 14×14, 196 patch)        |
| Hidden dim          | 768                                  |
| Attention heads     | 12                                   |
| Encoder layers      | 12                                   |
| Classification head | `nn.Linear(768, 2)`                  |

## Pembagian Data

| Split | Proporsi | Ukuran |
| ----- | -------- | ------ |
| Train | 70%      | 10.500 |
| Val   | 10%      | 1.500  |
| Test  | 20%      | 3.000  |

Split stratifikasi berdasarkan label. Split: 20% test dulu, lalu 12.5% dari sisanya untuk val.

## Preprocessing & Augmentasi

- Resize 224×224 (BICUBIC)
- Normalisasi ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- Augmentasi train: random horizontal flip, ColorJitter (brightness/contrast/saturation 0.1), random rotation ±10°
- Val/Test tanpa augmentasi
- `Batch size = 64`

## Hyperparameter

| Konfigurasi    | Nilai                                       |
| -------------- | ------------------------------------------- |
| Optimizer      | AdamW                                       |
| Learning rate  | 4e-5                                        |
| Weight decay   | 0.01                                        |
| Loss           | CrossEntropyLoss dengan label smoothing 0.1 |
| Scheduler      | CosineAnnealingLR (T_max=25, eta_min=1e-6)  |
| Max epochs     | 25                                          |
| Early stopping | patience=3 (berdasarkan val_loss)           |

## Metodologi

1. **Setup & Instalasi** — timm, PyTorch, torchvision, sklearn
2. **Ekstraksi Dataset** — path AI dan real
3. **EDA** — distribusi kelas, sampel gambar, analisis resolusi
4. **Split & Preprocessing** — stratified split, transformasi + augmentasi
5. **Inisialisasi ViT** — transfer learning, ganti head jadi 2 kelas
6. **Pelatihan** — training loop + cosine annealing + early stopping, save best model
7. **Evaluasi** — accuracy, precision, recall, F1, AUC-ROC, confusion matrix, ROC curve, prediksi sampel
8. **Baseline Comparison** — ResNet-50 dan EfficientNet-B0 dengan setup identik
9. **Robustness Testing** — JPEG compression (Q=90..10), variasi resolusi (192..64), Gaussian noise (σ=0.01..0.20) pada 300 sampel test
10. **Attention Rollout** — heatmap fokus model
11. **Ringkasan Akhir**

## Hasil Evaluasi (Test Set)

| Metrik | Nilai |
|---|---|
| Accuracy | 99.23% |
| Precision | 99.00% |
| Recall | 99.47% |
| F1-Score | 99.24% |
| AUC-ROC | 99.96% |

## Demo Interaktif

App Streamlit untuk klasifikasi foto wajah: upload gambar → label Real/AI-Generated + probabilitas. 3 model pilihan (EfficientNet-B0, ViT-Base/16, ResNet-50), weight dimuat dari Hugging Face hub.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Deploy ke Streamlit Community Cloud (gratis): push repo GitHub, hubungkan di streamlit.io.

## Struktur Notebook

```
real-or-fake-faces-with-vit.ipynb
├── 1. Setup & Instalasi
├── 2. Ekstraksi Dataset
├── 3. Eksplorasi Data (EDA)
├── 4. Pembagian Dataset & Preprocessing
├── 5. Inisialisasi Model ViT
├── 6. Konfigurasi Hyperparameter dan Pelatihan
├── 7. Evaluasi Model
├── 8. Baseline Comparison (ResNet-50, EfficientNet-B0)
├── 9. Robustness Testing + Attention Rollout
└── 10. Ringkasan Akhir
```

## Teknologi

- Python
- PyTorch + torchvision
- timm (ViT-Base/16, EfficientNet-B0)
- scikit-learn (metrik evaluasi)
- pandas, matplotlib, seaborn
- PIL

Dijalankan di **Kaggle** (path dataset `/kaggle/input/...`, output `/kaggle/working/...`).

## File Output

- `best_vit_model.pth`, `best_resnet50.pth`, `best_efficientnet_b0.pth`
- `evaluation_results.csv`
- `training_curves.png`, `confusion_matrix.png`, `roc_curve.png`
- `baseline_comparison.png`, `robustness_analysis.png`, `attention_maps.png`

## Catatan

- Seed `42` untuk reproducibility
- Notebook dijalankan di Kaggle dengan GPU (CUDA)
- Robustness testing memakai subset 300 gambar dari test set untuk kecepatan
