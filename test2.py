
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *
import cv2

plt.ion()
# fig, axes = plt.subplots(2, 2, num='Metrics')
cam = cv2.VideoCapture(0)

cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack(
        [transforms.ToTensor()(crop) for crop in crops])),
])


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


class_names = ['Angry', 'Disgust', 'Fear',
               'Happy', 'Sad', 'Surprise', 'Neutral']
ind = 0.1+0.6*np.arange(len(class_names))
width = 0.4       # the width of the bars: can also be len(x) sequence
color_list = ['red', 'orangered', 'darkorange',
              'limegreen', 'darkgreen', 'royalblue', 'navy']

net = VGG('VGG19')
checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()
plt.rcParams['figure.figsize'] = (13.5, 5.5)
fig, axes = plt.subplots(1, 2)

while True:
    _, raw_img = cam.read()
    raw_img = cv2.flip(raw_img, 1)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    raw_tensor = transforms.ToTensor()(raw_img)
    print(raw_tensor.size())
    raw_test = raw_tensor.permute(1, 2, 0).data.numpy()*255
    raw_test = raw_test.astype(np.uint8)

    # print(type(raw_test), raw_test.shape, raw_test.dtype)
    # print(type(raw_img), raw_img.shape, raw_img.dtype)
    # print(raw_img)
    # print(raw_test)
    # print((raw_img == raw_test).all())
    gray = rgb2gray(raw_test)
    gray_gt = rgb2gray(raw_img)
    # print((gray == gray_gt).all())
    # print(gray)
    # print(gray_gt)
    # import sys
    # sys.exit()
    gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)

    img = gray[:, :, np.newaxis]

    with torch.no_grad():
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        inputs = transform_test(img)

        ncrops, c, h, w = np.shape(inputs)

        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.cuda()
        inputs = Variable(inputs)
        outputs = net(inputs)
        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

        score = F.softmax(outputs_avg, dim=-1)
        print(score[3])
        _, predicted = torch.max(outputs_avg.data, 0)

    axes[0].clear()
    axes[1].clear()

    axes[0].imshow(raw_img / raw_img.max())
    for i in range(len(class_names)):
        axes[1].bar(ind[i], score.data.cpu().numpy()
                       [i], width, color=color_list[i])
    axes[1].set_xticks(ind)
    axes[1].set_xticklabels(class_names, rotation=90, ha="left")
    # axes[0, 1].imshow(image / image.max())
    # axes[1, 0].imshow(ref_ldmk / ref_ldmk.max())
    # axes[1, 1].imshow(ref_img / ref_img.max())
    # axes[0, 0].text(0, 0, f"{np.linalg.norm(ref_ldmk_pts-landmark_pts)}")
    fig.canvas.draw()
    fig.canvas.flush_events()

    print("The Expression is %s" %
          str(class_names[int(predicted.cpu().numpy())]))

cam.release()


# print("torch version : ", torch.__version__)
# print("Device : ", DEVICE)
# # torch.autograd.set_detect_anomaly(True)

# embeddings, paramWeights, paramBias = emb(context)
# synth_im = gen(gt_landmarks,  paramWeights, paramBias)
