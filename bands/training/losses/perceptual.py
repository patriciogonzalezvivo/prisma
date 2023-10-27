import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg import VGG19, VGG16

class Perceptual16Loss(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(Perceptual16Loss, self).__init__()
        self.vgg = VGG16()
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def calculate_pl(self, x, y):
        feat_output = self.vgg(x)
        feat_gt = self.vgg(y)

        content_loss = 0.0

        for i in range(3):
            content_loss += self.criterion(feat_output[i], feat_gt[i])
        return content_loss.to(device=x.device)
    
    def compute_gram(self, x):
        b, c, h, w = x.size()
        f = x.view(b, c, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * c)
        return G

    def calc_style(self, x, y):
        feat_output = self.extractor(x)
        feat_gt = self.extractor(y)

        style_loss = 0.0

        for i in range(3):
            style_loss += self.criterion(
                self.compute_gram(feat_output[i]), self.compute_gram(feat_gt[i]))
        return style_loss

class Perceptual19Loss(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(Perceptual19Loss, self).__init__()
        self.vgg = VGG19()
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def calculate_pl(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        prefix = [1, 2, 3, 4, 5]
        for i in range(5):
            content_loss += self.weights[i] * self.criterion(
                x_vgg[f'relu{prefix[i]}_1'], y_vgg[f'relu{prefix[i]}_1'])
        return content_loss.to(device=x.device)
    
    def compute_gram(self, x):
        b, c, h, w = x.size()
        f = x.view(b, c, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * c)
        return G

    def calc_style(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        style_loss = 0.0
        prefix = [2, 3, 4, 5]
        posfix = [2, 4, 4, 2]
        for pre, pos in list(zip(prefix, posfix)):
            style_loss += self.criterion(
                self.compute_gram(x_vgg[f'relu{pre}_{pos}']), self.compute_gram(y_vgg[f'relu{pre}_{pos}']))
        return style_loss
    