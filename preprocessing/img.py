import numpy as np
import torch
from preprocessing.padding import SquarePadding, get_padding
from torchvision import transforms
import cv2
from PIL import Image
import utils.img
import sys


# img shape: (height, width, channels)
def prepare_image(img):
    input_res = 256
    height, width = img.shape[0:2]
    c = np.array((width/2, height/2))
    s = max(height, width)/200

    cropped = utils.img.crop(img, c, s, (input_res, input_res))
    cropped = cropped / 255
    inp = torch.from_numpy(cropped.copy()) # returns shape  [256, 256, 3]

    inp = inp.permute(2, 0, 1) # change shape (h, w, c) to (c, h, w )

    # inp shape: [3, 256, 256]
    inp = inp.type(torch.FloatTensor).unsqueeze(dim=0) # add batch dimension
    return inp, c, s



def PIL_to_opencv(img):
    # use numpy to convert the pil_image into a numpy array
    numpy_image=np.array(img)

    # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that
    # the color is converted from RGB to BGR format
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return opencv_image


def opencv_to_PIL(img):
    # convert from openCV2 to PIL. Notice the COLOR_BGR2RGB which means that
    # the color is converted from BGR to RGB
    color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image=Image.fromarray(color_coverted)
    return pil_image



def prepare_image_old(img):
    img = opencv_to_PIL(img)
    orig_width, orig_height = img.size
    l_pad, t_pad, _, _ = get_padding(img)

    transform = transforms.Compose([
            SquarePadding(),
            transforms.Resize((256, 256)),
    ])

    img = transform(img)
    convert_to_tensor = transforms.ToTensor()
    img = convert_to_tensor(img) # returns shape (C x H x W)
    img = img/255
    return img.unsqueeze(0).type(torch.FloatTensor), l_pad, t_pad, orig_height, orig_width
