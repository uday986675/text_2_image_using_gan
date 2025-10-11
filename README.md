# Text-to-Image GAN Pipeline (Internship Project)

This project implements a **comprehensive text-to-image generation pipeline** that integrates:
- Text preprocessing & embedding
- GAN-based image synthesis
- Evaluation with a simple CNN classifier

### 📘 Overview
I simulate a text-to-image problem using synthetic geometric shapes described by color and shape (e.g., "red circle").  
The model learns to generate corresponding images from text embeddings.

### 🧠 Components
- **Text Preprocessing:** Tokenization and embedding
- **Generator / Discriminator:** Conditional GAN (DCGAN-style)
- **Training:** Adversarial training loop with visual outputs
- **Evaluation:** Simple classifier accuracy on generated samples

### 🧩 Folder Structure
```
text_to_image_internship_project/
├── README.md
├── requirements.txt
├── src/
│   ├── dataset.py
│   └── models.py
├── notebooks/
│   └── text_to_image_internship.ipynb
└── outputs/
```
### 🖼️ Example Captions
```
red circle
blue triangle
green square
```

### ⚙️ How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/text_to_image_internship.ipynb
```
Results will be saved in the `outputs/` folder.
