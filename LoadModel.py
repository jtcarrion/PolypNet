import os, cv2
import pandas as pd
import io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision.io import read_image
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFilter

print(os.getcwd())
os.chdir('C:/Users/16023/Desktop/BMI_540_Project')

img = ['./OG_test/55.tif', './OG_test/37.tif']
mask = ['./GT_test/55.tif', './GT_test/37.tif']

data = pd.DataFrame(columns=[['img_path'], ['mask_path']])
data['img_path'] = img
data['mask_path'] = mask

img_size = (288, 384, 3)
num_classes = 1
batch_size = 16
class_names = ['background', 'polyp']
class_rgb_values = [[0, 0, 0], [255, 255, 255]]


def mask_up(x, y):
    im1 = Image.open(x)
    im2 = Image.open(y)

    m = Image.new("L", im1.size, 128)
    im = Image.composite(im1, im2, m)
    return im


def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


class EndoscopyPrediction(torch.utils.data.Dataset):
    """CVC-ClinicDB Endoscopic Colonoscopy Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        df (str): DataFrame containing images / labels paths
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            df,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.image_paths = data['img_path'].values.tolist()
        self.mask_paths = data['mask_path'].values.tolist()
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read images and masks
        image1 = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask1 = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image1, mask=mask1)
            image1, mask1 = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image1, mask=mask1)
            image1, mask1 = sample['image'], sample['mask']

        return image1, mask1

    def __len__(self):
        # return length of
        return len(self.image_path)


# Center crop padded image / mask to original image dims
def crop_image(pic, true_dimensions):
    return album.CenterCrop(p=1, height=true_dimensions[0], width=true_dimensions[1])(image=pic)


ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = class_names
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# load best saved model checkpoint from the current run
if os.path.exists('./UNet.pth'):
    model = torch.load('./UNet.pth', map_location=torch.device('cpu'))
    print('Loaded UNet model from this run.')

x = r'./OG_test/55_test.png'
y = r'./GT_test/55_test.png'
z = mask_up(x,y)
z.show()


'''
im1 = Image.open(r'./OG_test/55_test.png')
im2 = Image.open(r'./GT_test/55_test.png')

mask = Image.new("L", im1.size, 128)
im = Image.composite(im1, im2, mask)
test_transform = [album.PadIfNeeded(min_height=288, min_width=384, always_apply=True, border_mode=0)]

f = album.Compose(test_transform)

#print(model.eval())

#img = Image.open(img)
#x1 = torch.tensor(img, dtype=torch.PngImageFile).to(torch.device("cpu")).unsqueeze(0)
#x1 = x1[:3]

#image = cv2.cvtColor(cv2.imread(r"./OG_test/55.tif"), cv2.COLOR_BGR2RGB)
#print(image.shape)
#print(image[0:1].shape)

#lol = f(image=image[0:1])
#x = lol['image']
#y = list(x)
#z = np.array(y)
#print(z.shape)

#x_tensor = torch.from_numpy(image).to(torch.device("cpu")).unsqueeze(0)

#test_img = EndoscopyPrediction(data,
#                               augmentation=get_validation_augmentation(),
#                               preprocessing=get_preprocessing(preprocessing_fn),
#                               )
#lol = test_img[0]

#lol = read_image(img)
#print(lol.shape)
#pad = nn.ZeroPad2d((0,1,1,0))
# pad the input tensor
#output = pad(lol)
#print(output.shape)

#pred_mask = model(lol)
#pred_mask = pred_mask.detach().squeeze().cpu().numpy()
# Convert pred_mask from `CHW` format to `HWC` format
#pred_mask = np.transpose(pred_mask,(1,2,0))


'''