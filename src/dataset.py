import torch, random, numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

SHAPES = ['circle','square','triangle']
COLORS = ['red','green','blue','yellow','magenta','cyan']

def draw_shape(shape, color, size=64):
    img = Image.new('RGB', (size,size), (255,255,255))
    draw = ImageDraw.Draw(img)
    pad = int(size*0.15)
    bbox = [pad, pad, size-pad, size-pad]
    color_map = {
        'red':(230,25,75), 'green':(60,180,75), 'blue':(0,130,200),
        'yellow':(255,225,25), 'magenta':(240,50,230), 'cyan':(70,240,240)
    }
    c = color_map[color]
    if shape=='circle': draw.ellipse(bbox, fill=c)
    elif shape=='square': draw.rectangle(bbox, fill=c)
    elif shape=='triangle':
        x0,y0,x1,y1 = bbox
        pts = [(size/2,y0),(x1,y1),(x0,y1)]
        draw.polygon(pts, fill=c)
    return img

class ShapesDataset(Dataset):
    def __init__(self, n_images=2000, img_size=64):
        self.records = []
        for _ in range(n_images):
            shape = random.choice(SHAPES)
            color = random.choice(COLORS)
            caption = f"{color} {shape}"
            self.records.append({'shape':shape,'color':color,'caption':caption})
        self.img_size = img_size

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = draw_shape(rec['shape'], rec['color'], self.img_size)
        arr = torch.tensor(np.array(img).transpose(2,0,1)/127.5-1.0, dtype=torch.float32)
        return arr, rec['caption']
