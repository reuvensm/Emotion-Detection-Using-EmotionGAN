import os
from pathlib import Path
import sys

import dlib
import matplotlib.pyplot as plt
import torch
import cv2
from imutils.face_utils import rect_to_bb
import numpy as np

from ..models.model_linear_2d import Generator as Generator_l2
from ..models.model_gaussian_2d import Generator as Generator_g2


class GANmut:

    def __init__(self, G_path, model='linear', g_conv_dim=64, c_dim=7, g_repeat_num=6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model

        if self.model == 'linear':
            self.G = Generator_l2(self.device, g_conv_dim, c_dim, g_repeat_num)

        elif self.model == 'gaussian':
            self.G = Generator_g2(self.device, g_conv_dim, c_dim, g_repeat_num)

        else:
            raise ValueError("choose either model='linear' or model='gaussian'")

        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.G.to(self.device)
        self.detector = dlib.get_frontal_face_detector()

    def emotion_edit(self, img_path, x=None, y=None, theta=None, rho=None, save=False, save_dir='../edited_images', save_name=''):

        if self.model == 'linear':
            assert (rho is not None) or (theta is not None), 'if model is linear you must provide rho and theta'
        else:
            assert (x is not None) and (y is not None), 'if model is gaussian you must provide x and y'
        if isinstance(img_path, str):
            img = cv2.imread(img_path, 1)  # BGR
        else:
            img = img_path
        img_rgb = img[:, :, [2, 1, 0]]
        # plt.title('Original Image')
        # plt.imshow(img_rgb)
        HEIGHT = img.shape[0]
        WIDTH = img.shape[1]

        # extract face
        det = self.detector(img, 1)
        if len(det) == 0:  # Face not found
            # print("Face not found!!")
            return img, False
        else:
            det = det[0]
        (xx, yy, w, h) = rect_to_bb(det)
        xx = max(xx, 0)
        yy = max(yy, 0)
        if yy + h >= HEIGHT:
            h = HEIGHT - yy
        if xx + w >= WIDTH:
            w = WIDTH - xx
        # print("(xx, yy, w, h)", (xx, yy, w, h))
        try:
            face = cv2.resize(img[yy:yy + h, xx:xx + w], (128, 128))
        except:
            # print("Face not found due to resize problem!!")
            return img, False


        #plt.figure()
        #plt.title('Detected face')
        #plt.imshow(face[:, :, [2, 1, 0]])

        # adapt image format for G
        face = face.transpose((2, 0, 1))  # [H,W,C] --> [C,H,W]
        face = (face / 255.0 - 0.5) / 0.5  # normalize to [-1, 1]
        face = torch.from_numpy(face).float().unsqueeze(0).to(self.device)

        # edit emotion

        with torch.no_grad():

            if self.model == 'linear':
                mode = 'manual_selection'
                expr = (torch.tensor([np.cos(theta), np.sin(theta)]) * rho).to(self.device).float()
                face_g = self.G(face, None, None, mode=mode, manual_expr=expr)[0][0, [2, 1, 0], :, :] / 2 + 0.5
            else:
                expr = torch.Tensor([x, y]).unsqueeze(0).to(self.device)
                face_g = self.G(face, expr)[0][0, [2, 1, 0], :, :] / 2 + 0.5

        face_g = face_g.transpose(0, 2).transpose(0, 1).detach().cpu().numpy()
        
        temp_g = face_g 
        
#         plt.figure()
#         plt.title('Edited face')
#         plt.imshow(face_g)

        # insert edited face in original image
#         print(w, h)
        if yy + h >= HEIGHT:
            yy = HEIGHT - h
        if xx + w >= WIDTH:
            xx = WIDTH - w
        
        img_rgb[yy:yy + h, xx:xx + w] = cv2.resize(face_g, (w, h)) * 255

#         plt.figure()
#         plt.title('Edited image')
#         plt.imshow(img_rgb)

        if save:
            #save_dir = "../edited_images"
            #if(self.model == 'linear'):
            #    save_dir = "../edited_images_linear"
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            org_filename = 'image.png' if not isinstance(img_path, str) else os.path.split(img_path)[-1]
            if save_name == '' and self.model == 'linear':
                img_name = 'theta_{:0.2f}_rho_{:0.2f}'.format(theta, rho) + org_filename
            elif save_name == '':
                img_name = 'x_{:0.2f}_y_{:0.2f}'.format(x, y) + org_filename
            else:
                img_name = save_name

            img_name = os.path.join(save_dir, img_name)
            plt.imsave(img_name, img_rgb)
#             print(f'edited image saved in {img_name}')
            return img_name, True
        return img_rgb, True
