import time
import cv2
import torch
from torchvision import transforms
from PIL import Image
import sys
import matplotlib.pyplot as plt

from models.stacked_hourglass import PoseNet, SGSC_PoseNet


def load_model(arch='SGSC'):

    nstack = 4
    inp_dim = 144
    oup_dim = 16
    checkpoint_path = 'models/SGSC_model_weights.tar'

    if arch == 'SHG':
        nstack = 8
        inp_dim = 256
        checkpoint_path = 'models/SHG_model_weights.tar'

    if arch == 'SHG':
        net = PoseNet(nstack, inp_dim, oup_dim)
    else:
        net = SGSC_PoseNet(nstack, inp_dim, oup_dim)


    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net
