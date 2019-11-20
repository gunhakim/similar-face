import pickle
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import cv2

import data

model = models.vgg16(pretrained=True).features
model.eval()
trans = transforms.ToTensor()

img_path = "/Users/kimgunha/code/ml_dragon/projects/test/미나.jpg"

with torch.no_grad():
    try:
        img = data.img_transform_if(img_path)
    
    except:
        print("사람이 아닙니다")
        exit()
    img = img.unsqueeze(0)
    out = model(img)
    out = out.squeeze()
    out = out.mean(dim=(1, 2)).numpy()

with open("/Users/kimgunha/code/ml_dragon/projects/features.pkl", "rb") as f:
    features, label = pickle.load(f)


nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='cosine').fit(features)
distances, indices = nbrs.kneighbors(out.reshape(1, -1))

# print(indices)
# print(distances)

target = cv2.imread(img_path)
target = data.img_crop(target)
output = cv2.imread("/Users/kimgunha/code/ml_dragon/projects/export/%s.jpg"%(indices.squeeze()))

result1 = cv2.addWeighted(target, 1, output, 0, 0)
result2 = cv2.addWeighted(target, 0.67, output, 0.33, 0)
result3 = cv2.addWeighted(target, 0.33, output, 0.67, 0)
result4 = cv2.addWeighted(target, 0, output, 1, 0)
result = cv2.hconcat([result1, result2, result3, result4])

print(label[indices.squeeze()])

cv2.imshow('image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
