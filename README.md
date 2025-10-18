

Text-to-Image GAN with BERT Embeddings
A PyTorch implementation of a Generative Adversarial Network (GAN) that generates flower images from text descriptions using BERT embeddings.

Overview
This project implements a conditional GAN that generates 64x64 flower images based on text descriptions. The model uses BERT to encode text descriptions into embeddings, which are then used to condition both the generator and discriminator during training.

Features
Conditional GAN Architecture: Generator and discriminator conditioned on text embeddings

BERT Text Encoding: Uses BERT-base-uncased for text feature extraction

Oxford Flowers Dataset: Trained on the Oxford 102 Category Flower Dataset

Training Monitoring: Includes FID and Inception Score evaluation

Visualization: Saves generated samples after each epoch

Model Architecture
Generator
Input: Noise vector (100-dim) + BERT embedding (768-dim)

Architecture: Fully connected layer + transposed convolutional layers

Output: 64x64 RGB image

Discriminator
Input: 64x64 RGB image + BERT embedding (768-dim)

Architecture: Convolutional layers + fully connected layer

Output: Binary classification (real/fake)

Requirements
bash
pip install torch torchvision transformers tqdm pillow scipy numpy matplotlib torchmetrics
Dataset Setup
Download the Oxford 102 Category Flower Dataset

Update the data paths in the configuration:

DATA_PATH: Path to flower images folder

LABEL_PATH: Path to imagelabels.mat file

Usage
Training
python
# Configuration
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10
LR = 2e-4
Z_DIM = 100
EMBED_DIM = 768

# Initialize models
gen = Generator(Z_DIM, EMBED_DIM).to(device)
disc = Discriminator(EMBED_DIM).to(device)

# Training loop
for epoch in range(EPOCHS):
    # Training steps for discriminator and generator
    # Saves generated samples each epoch
Evaluation
The model includes evaluation using:

Frechet Inception Distance (FID)

Inception Score

File Structure
text
Text_to_Image_GAN_Enhanced/
├── Text_to_Image_GAN_Enhanced.ipynb  # Main training notebook
├── README.md                         # This file
└── gen_samples_epoch*.png           # Generated samples (created during training)
Results
Generated flower images are saved as gen_samples_epoch{epoch}.png

Training losses for generator and discriminator are printed each epoch

Quantitative evaluation using FID and Inception Score

Key Components
FlowersDataset: Custom dataset class that loads images and generates text embeddings

BERT Integration: Uses pre-trained BERT model for text feature extraction

Conditional GAN: Text-conditioned image generation

Evaluation Metrics: Standard GAN evaluation metrics

Notes
The model uses simple text descriptions based on flower class labels

Images are normalized to [-1, 1] range

Training uses Adam optimizer with standard GAN parameters

Generated samples are denormalized for visualization

Future Improvements
Use more descriptive text captions

Implement attention mechanisms

Add data augmentation

Experiment with different GAN architectures (DCGAN, StyleGAN)

Incorporate CLIP for better text-image alignment

Citation
If you use this code in your research, please cite:

bibtex
@misc{text2image_gan_flowers,
  author = {Your Name},
  title = {Text-to-Image GAN with BERT Embeddings},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/text-to-image-gan}}
}

Explain
