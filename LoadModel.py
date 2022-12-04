import os, cv2
import pandas as pd
import torch
import numpy as np
import albumentations as album
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from PIL import Image, ImageDraw, ImageFilter

img_size = (288, 384, 3)
num_classes = 1
class_names = ['background', 'polyp']
class_rgb_values = [[0, 0, 0], [255, 255, 255]]

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = class_names
ACTIVATION = 'sigmoid' 


def crop_image(pic, true_dimensions):
    return album.CenterCrop(p=1, height=true_dimensions[0], width=true_dimensions[1])(image=pic)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def predict(path):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    image = crop_image(image, [288, 384])
    image = to_tensor(image)
    pred_mask = model(image)
    pred_mask = pred_mask.detach().squeeze().cpu().numpy()
    pred_mask = np.transpose(pred_mask,(1,2,0))
    return pred_mask


def mask_up(x):
    im1 = Image.open(x)
    im2 = predict(x)
    m = Image.new("L", im1.size, 128)
    im3 = Image.composite(im1, im2, m)
    return im3


# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

# load best saved model checkpoint from the current run
if os.path.exists('./UNet.pth'):
    model = torch.load('./UNet.pth', map_location=torch.device('cpu'))
    print('Loaded UNet model.')
