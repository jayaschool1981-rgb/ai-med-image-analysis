"""Generate a tiny dummy chest X-ray dataset with two classes:
- normal: lighter backgrounds with mild noise
- pneumonia: add circular 'opacity' blobs
This is for pipeline testing only. NOT medically representative.
"""
import os, random
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

def gen_normal(w=512, h=512):
    base = np.clip(np.random.normal(loc=200, scale=18, size=(h, w)), 0, 255).astype(np.uint8)
    img = Image.fromarray(base, mode="L").filter(ImageFilter.GaussianBlur(1.5))
    return img

def gen_pneumonia(w=512, h=512):
    img = gen_normal(w, h).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    for _ in range(random.randint(2,5)):
        r = random.randint(40, 120)
        x = random.randint(r, w-r)
        y = random.randint(r, h-r)
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(120,120,120,random.randint(80,140)))
    img = img.convert("L").filter(ImageFilter.GaussianBlur(2.0))
    return img

def write_split(root, split="train", n_per_class=12):
    for cls, fn in [("NORMAL", gen_normal), ("PNEUMONIA", gen_pneumonia)]:
        d = os.path.join(root, split, cls)
        os.makedirs(d, exist_ok=True)
        for i in range(n_per_class):
            img = fn()
            img = img.resize((1024,1024))
            img.save(os.path.join(d, f"{cls.lower()}_{i:03d}.png"))

if __name__ == "__main__":
    root = "data/samples"
    # structure: train/val/test with fewer images for val/test
    write_split(root, "train", 20)
    write_split(root, "val", 6)
    write_split(root, "test", 6)
    print("Dummy dataset created at data/samples/{train,val,test}/{NORMAL,PNEUMONIA}")
