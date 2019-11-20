import glob
import os
import pickle
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

import data

model = models.vgg16(pretrained=True).features
model.eval()
trans = transforms.ToTensor()

img_dir = "/Users/kimgunha/code/ml_dragon/projects/train/"
images = glob.glob(img_dir + "*.*")

features = []
labels = []
with torch.no_grad():
    for i, img in enumerate(images):
        name = os.path.basename(img)
        
        label = name.split(".")[0]
        if label[-1].isdigit():
            label = label[:-1]
        print(label)

        img = data.img_transform(img, i)
        img = img.unsqueeze(0)
        out = model(img)
        out = out.squeeze()
        out = out.mean(dim=(1, 2)).numpy()
        
        labels.append(label)
        features.append(out)

print(labels)
print(np.array(features).shape)

with open("/Users/kimgunha/code/ml_dragon/projects/features.pkl", "wb") as f:
    pickle.dump([np.array(features), labels], f)