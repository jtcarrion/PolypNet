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

print(os.getcwd())
os.chdir('C:/Users/16023/Desktop/BMI_540_Project')

img_size = (288, 384, 3)
num_classes = 1
batch_size = 16
class_names = ['background', 'polyp']
class_rgb_values = [[0, 0, 0], [255, 255, 255]]


# helper function for data visualization
def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]);
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()


def get_validation_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.PadIfNeeded(min_height=288, min_width=384, always_apply=True, border_mode=0),
    ]
    return album.Compose(test_transform)


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
        self.image_paths = self
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read images and masks
        image1 = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
       # mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)

        # one-hot-encode the mask
        #mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image1, mask=image1)
            image1, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image1, mask=image1)
            image1, mask = sample['image'], sample['mask']

        return image1

    def __len__(self):
        # return length of
        return len(self.image_paths)


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

#preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# load best saved model checkpoint from the current run
if os.path.exists('./UNet.pth'):
    model = torch.load('./UNet.pth', map_location=torch.device('cpu'))
    print('Loaded UNet model from this run.')

test_transform = [album.PadIfNeeded(min_height=288, min_width=384, always_apply=True, border_mode=0)]

f = album.Compose(test_transform)

#print(model.eval())

img = './OG_test/55_test.png'

#image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
#lol = f(image=image)
#x = lol['image']
#y = list(x)
#z = np.array(y)

#test_img = EndoscopyPrediction(img)
#lol = test_img[0]


lol = read_image(img)
pad = nn.ZeroPad2d(1)

# pad the input tensor
output = pad(lol)
torch.reshape(output, (288, 384))
#x_tensor = torch.from_numpy(lol[:,:]).to(torch.device("cpu")).unsqueeze(0)

pred_mask = model(output)
pred_mask = pred_mask.detach().squeeze().cpu().numpy()
# Convert pred_mask from `CHW` format to `HWC` format
pred_mask = np.transpose(pred_mask,(1,2,0))


visualize(
        original_image = img,
        predicted_mask = pred_mask
    )