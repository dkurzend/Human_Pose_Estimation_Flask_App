import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
import cv2
import utils.img
import sys


# input shape: (batch_size, 16, 64, 64)
def get_keyppoints(heatmaps, threshold):
    batch_size = heatmaps.shape[0]
    num_keypoints = heatmaps.shape[1]
    keypoints = np.zeros((batch_size, num_keypoints, 2)) # shape (16, 16, 2)

    for batch in range(batch_size):
        for kp in range(num_keypoints):
            hm = heatmaps[batch, kp, :,:]
            if torch.max(hm).item() > float(threshold):
                # flatten the thensor and get indices of 2 highest values
                v, i = torch.topk(hm.flatten(), 2)

                # convert back to 2d matrix
                # keypoint[0] contains height or y axis
                # keypoint[1] contains width or x axis
                top_k = np.array(np.unravel_index(i.numpy(), hm.shape)).T # top_K has shape [2, 2]
                keypoint = torch.tensor(top_k[0]).type(torch.FloatTensor)

                # move y one quarter pixel towards next highest pixel
                if top_k[1, 0] < keypoint[0]:
                    keypoint[0] -= 0.25
                if top_k[1, 0] > keypoint[0]:
                    keypoint[0] += 0.25

                # move x one quarter pixel towards next highest pixel
                if top_k[1, 1] < keypoint[1]:
                    keypoint[1] -= 0.25
                if top_k[1, 1] > keypoint[1]:
                    keypoint[1] += 0.25

                # we return the keypoints as (x, y) = (width, height) coordinates
                keypoints[batch, kp, 0] = keypoint[1]
                keypoints[batch, kp, 1] = keypoint[0]

            else:
                keypoints[batch, kp, 0] = 0
                keypoints[batch, kp, 1] = 0

    return keypoints






# input shape: (batch_size, 16, 64, 64)
def get_keyppoints_without_adjustment(heatmaps):
    batch_size = heatmaps.shape[0]
    num_keypoints = heatmaps.shape[1]
    keypoints = np.zeros((batch_size, num_keypoints, 2)) # shape (16, 16, 2)

    for batch in range(batch_size):
        for kp in range(num_keypoints):
            hm = heatmaps[batch, kp, :,:]

            # keypoint[0] contains height or y axis
            # keypoint[1] contains width or x axis
            keypoint = (hm==torch.max(hm)).nonzero()[0]

            # we return the keypoints as (x, y) = (width, height) coordinates
            keypoints[batch, kp, 0] = keypoint[1]
            keypoints[batch, kp, 1] = keypoint[0]

    return keypoints


# keypoints: (batch_size, 16, 2)
# img: [1, 3, 256, 256]
def post_process_keypoints(keypoints, img, c, s, input_res):
    height, width = img.shape[2:]
    center = (width/2, height/2)
    scale = max(height, width)/200
    res = (input_res, input_res)

    mat_ = utils.img.get_transform(center, scale, res)[:2]
    mat = np.linalg.pinv(np.array(mat_).tolist() + [[0,0,1]])[:2]

    # transform to 256x256 resolution
    keypoints[:,:,:2] = utils.img.kpt_affine(keypoints[:,:,:2] * 4, mat)
    preds = np.copy(keypoints)

    # transform each keypoint to orig resolution
    for j in range(preds.shape[1]):
        preds[0,j,:2] = utils.img.transform(preds[0,j,:2], c, s, res, invert=1)
    return preds # shape: [1, 16, 2]



# keypoints shape = (16, 2)
# kp[0] = x axis = width
# kp[1] = y axis = height
def draw_keypoints(image, keypoints, radius=12):
    transform_to_pillow = transforms.ToPILImage()
    image = transform_to_pillow(image)
    draw = ImageDraw.Draw(image)
    num_keypoints = keypoints.shape[0]

    for i in range(num_keypoints):
        draw.ellipse((keypoints[i, 0]-radius, keypoints[i, 1]-radius, keypoints[i, 0]+radius, keypoints[i, 1]+radius), fill = 'red', outline ='green')

    convert_to_tensor = transforms.ToTensor()
    image = convert_to_tensor(image) # returns shape (C x H x W)
    return image



# keypoints shape = (16, 2)
# kp[0] = x axis = width
# kp[1] = y axis = height
# image is a cv2 image
def draw_cv2_keypoints(image, keypoints, radius=12, mode = 2):
    num_keypoints = keypoints.shape[0]

    for i in range(num_keypoints):
        if (mode == 0):
            draw_cv2_linesBetweenKeypoints(image, keypoints)

        if(mode == 1):
            draw_text(image,keypoints)
        if(mode == 2):
            draw_cv2_linesBetweenKeypoints(image, keypoints)
            draw_text(image,keypoints)

    for i in range(num_keypoints):

        if (keypoints[i,0]>=1) and (keypoints[i,1]>=1):
            point = (int(keypoints[i, 0]), int(keypoints[i, 1]))
            image = cv2.circle(image, point, radius=radius, color=(0, 0, 255), thickness=-1)
    return image


#display of Keypointnams
def draw_text(image,keypoints):
    parts = ['rank', 'rkne', 'rhip',
                        'lhip', 'lkne', 'lank',
                        'pelv', 'thrx', 'neck', 'head',
                        'rwri', 'relb', 'rsho',
                        'lsho', 'lelb', 'lwri']
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 0.8
    thickness = 1
    for i in range(keypoints.shape[0]):
        if (keypoints[i,0]>=1) and (keypoints[i,1]>=1):
            org = (int(keypoints[i,0]),int(keypoints[i,1]))
            image = cv2.putText(image,parts[i],org, font,fontscale, (0,0,255), thickness, cv2.LINE_AA)


#lines between keypoints
def draw_cv2_linesBetweenKeypoints(image, keypoints):
    pairs = [[8,9],[7,8], [7,12],[7,13], [11,12],[13,14], [10,11],[14,15], [6,7],[2,6], [3,6],[1,2], [3,4],[0,1], [4,5]]
    for list in pairs:
        if (keypoints[list[0],0]>=1) and (keypoints[list[0],1]>=1) and (keypoints[list[1],0]>=1) and (keypoints[list[1],1]>=1):
            point = (int(keypoints[list[0], 0]), int(keypoints[list[0], 1]))
            point2 = (int(keypoints[list[1], 0]), int(keypoints[list[1], 1]))
            image = cv2.line(image, point,point2,(0,255,0), 3)




# keypoints has shape (1, 16, 3)
def downScaleKeypoints(keypoints, l_pad, t_pad, current_width, current_height, target_width, target_height):
    for i in range(np.shape(keypoints)[1]):
        if keypoints[0,i,0] > 0:
            # adjust keypoint coordinates to padding
            keypoints[0, i, 0] += l_pad
            keypoints[0, i, 1] += t_pad

            # transform keypoints to 64x64 resolution
            # kp[0] = x axis = width
            # kp[1] = y axis = height
            keypoints[0, i, 0] *= target_width/max(current_width, current_height)
            keypoints[0, i, 1] *= target_height/max(current_width, current_height)
    return keypoints


# keypoints has shape (batch_size, 16, 3)
def upScaleKeypoints(keypoints, l_pad, t_pad, current_width, current_height, target_width, target_height):
    batch_size = keypoints.shape[0]

    for dp in range(batch_size):
        for i in range(np.shape(keypoints)[1]):
            if keypoints[dp,i,0] > 0:

                # transform keypoints to 64x64 resolution
                # kp[0] = x axis = width
                # kp[1] = y axis = height
                keypoints[dp, i, 0] *= max(target_width, target_height)/current_width
                keypoints[dp, i, 1] *= max(target_width, target_height)/current_height

                # adjust keypoint coordinates to padding
                keypoints[dp, i, 0] -= l_pad
                keypoints[dp, i, 1] -= t_pad
            else:
                pass

    return keypoints
