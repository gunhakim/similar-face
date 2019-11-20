import numpy as np
import cv2, dlib
import torchvision.transforms as transforms
from PIL import Image

detector = dlib.get_frontal_face_detector()
totensor = transforms.ToTensor()
crop = transforms.functional.crop
resize = transforms.functional.resize


def img_transform(img_path, idx=0):
    img = cv2.imread(img_path)
    face = detector(img)[0]

    img = Image.open(img_path)
    img = crop(img, face.top(), face.left(), face.height(), face.width())
    img = resize(img, (150, 150))

    img.save("/Users/kimgunha/code/ml_dragon/projects/export/%s.jpg"%idx)
    return totensor(img)

def img_transform_if(img_path, idx=0):
    img = cv2.imread(img_path)
    face = detector(img)[0]

    img = Image.open(img_path)
    img = crop(img, face.top(), face.left(), face.height(), face.width())
    img = resize(img, (150, 150))

    # img.save("/Users/kimgunha/code/ml_dragon/projects/export/%s.jpg"%idx)
    return totensor(img)

def img_crop(img):
    face = detector(img)[0]

    img = img[face.top(): face.top()+face.height(), face.left(): face.left()+face.width()]
    img = cv2.resize(img, dsize=(150, 150))
    return img