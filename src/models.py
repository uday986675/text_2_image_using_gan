import torch
import torch.nn as nn
import torch.nn.functional as F

Z_DIM, TEXT_DIM = 100, 32

class SimpleTextEmbed(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.word_to_idx = {w:i for i,w in enumerate(vocab)}
        self.emb = nn.Embedding(len(vocab), TEXT_DIM)
    def forward(self, captions):
        ids = []
        for cap in captions:
            toks = cap.split()
            ids.append([self.word_to_idx[t] for t in toks])
        return self.emb(torch.tensor(ids)) .mean(1)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(Z_DIM+TEXT_DIM, 512*4*4), nn.ReLU(True))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64,3,3,1,1), nn.Tanh())
    def forward(self, z, txt):
        x = torch.cat([z,txt],1)
        x = self.fc(x).view(-1,512,4,4)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,64,4,2,1), nn.LeakyReLU(0.2,True),
            nn.Conv2d(64,128,4,2,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2,True),
            nn.Conv2d(128,256,4,2,1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2,True),
            nn.Conv2d(256,512,4,2,1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2,True))
        self.fc_img = nn.Linear(512*4*4, 1)
        self.fc_txt = nn.Linear(TEXT_DIM, 512*4*4)
    def forward(self, img, txt):
        h = self.conv(img).view(img.size(0), -1)
        proj = torch.sum(h * self.fc_txt(txt), 1, keepdim=True)
        return self.fc_img(h) + proj
