# coding=utf-8
'''
Author: husserl
License: Apache Licence
Software: VSCode
Date: 2023-07-12 11:46:28
LastEditors: husserl
LastEditTime: 2023-08-17 09:52:48
'''


from typing import Any
from PIL import Image, ImageFilter
import PIL.ImageEnhance as ImageEnhance
import random
import numpy as np



class ImgNormalize(object):
    def __init__(self, mean, std, *args, **kwargs):
        self.mean = mean
        self.std = std

    def __call__(self, im_dict):
        
        im = im_dict['image']
        img = np.asarray(im).astype(np.float32)
        img = (img - self.mean) / self.std
        im_dict['mean'] = self.mean
        im_dict['std'] = self.std
        im_dict['image'] = img
        return im_dict


class BottomCrop(object):
    def __init__(self, cropsize, *args, **kwargs):
        self.size = cropsize

    def __call__(self, im_dict):
        
        im = im_dict['image']
        W, H = self.size  # new
        w, h = im.size  # old

        if (W, H) == (w, h): 
            crop = 0, 0, w, h
            im_dict['crop'] = crop
            return im_dict
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            im_dict['rotation'] *= scale
        sw, sh = 0.5 * (w - W), 1.0 * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        im_dict['crop'] = crop
        im_dict['w'] = W
        im_dict['h'] = H
        im_dict['image'] = im.crop(crop)
        im_dict['translation'] -= crop[:2]

        return im_dict

class CenterCrop(object):
    def __init__(self, cropsize, *args, **kwargs):
        self.size = cropsize

    def __call__(self, im_dict):
        
        im = im_dict['image']
        W, H = self.size  # new
        w, h = im.size  # old

        if (W, H) == (w, h): 
            crop = 0, 0, w, h
            im_dict['crop'] = crop
            return im_dict
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            im_dict['rotation'] *= scale
        sw, sh = 0.5 * (w - W), 0.5 * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        im_dict['crop'] = crop
        im_dict['w'] = W
        im_dict['h'] = H
        im_dict['image'] = im.crop(crop)
        im_dict['translation'] -= crop[:2]

        return im_dict

class RandomCrop(object):
    def __init__(self, cropsize, *args, **kwargs):
        self.size = cropsize

    def __call__(self, im_dict):
        
        im = im_dict['image']
        W, H = self.size  # new
        w, h = im_dict['w'], im_dict['h']  # old

        if (W, H) == (w, h): 
            crop = 0, 0, w, h
            im_dict['crop'] = crop
            return im_dict
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            im_dict['rotation'] *= scale
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        # sw, sh = random.random() * (w - W), (1-0.5*random.random()) * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        im_dict['crop'] = crop
        im_dict['w'] = W
        im_dict['h'] = H
        im_dict['image'] = im.crop(crop)
        im_dict['translation'] -= crop[:2]

        return im_dict


class HorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_dict):
        if random.random() > self.p:
            im_dict['horizontalflip'] = False
            return im_dict
        else:
            im = im_dict['image']
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            im_dict['image'] = im
            A = np.asarray([[-1, 0], [0, 1]])
            b = np.asarray([im_dict['w'], 0])
            im_dict['rotation'] = np.dot(A, im_dict['rotation'])
            im_dict['translation'] = np.dot(A, im_dict['translation'])+b
            # cam_intrinsic
            im_dict['horizontalflip'] = True
            return im_dict

class VerticalFlip(object):
    def __init__(self, p=0.5, *arg, **kwargs):
        self.p = p
    
    def __call__(self, im_dict):
        if random.random() > self.p:
            im_dict['verticalflip'] = False
            return im_dict
        else:
            im = im_dict['image']
            im = im.transpose(Image.FLIP_TOP_BOTTOM)
            im_dict['image'] = im
            A = np.asarray([[1, 0], [0, -1]])
            b = np.asarray([0, im_dict['h']])
            im_dict['rotation'] = np.dot(A, im_dict['rotation'])
            im_dict['translation'] = np.dot(A, im_dict['translation'])+b
            # cam_intrinsic
            im_dict['verticalflip'] = True
            return im_dict


class RandomScale(object):
    def __init__(self, scale=1.0, *args, **kwargs):
        self.scale = scale

    def __call__(self, im_dict):
        im = im_dict['image']
        W, H = im.size
        # scale = np.random.uniform(min(self.scales), max(self.scales))
        w, h = int(W * self.scale), int(H * self.scale)
        im_dict['image'] = im.resize((w, h), Image.BILINEAR)
        im_dict['w'] = w
        im_dict['h'] = h
        # cam_intrinsic
        im_dict['scale'] = self.scale 
        im_dict['rotation'] *= self.scale
        return im_dict


class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im_dict):
        im = im_dict['image']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        im_dict['image'] = im
        return im_dict

class GaussBlur(object):
    """
    高斯噪声
    """
    def __init__(self, radius=(0,)):
        self.radius = radius
        
    def __call__(self, im_dict):
        im = im_dict['image']
        radius = random.choice(self.radius)
        im = im.filter(ImageFilter.GaussianBlur(radius=radius))
        im_dict['image'] = im
        return im_dict

class RotateImage(object):
    def __init__(self, rotate_boundary=[1,1]):
        self.rotate_boundary = rotate_boundary
        
    def __call__(self, im_dict):
        im = im_dict['image']
        rotate = np.random.uniform(*self.rotate_boundary)
        im = im.rotate(rotate)
        im_dict['image'] = im
        im_dict.update({'rotate_image': rotate})
        theta = rotate / 180 * np.pi
        A = np.asarray(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        b = np.asarray([im_dict['w'], im_dict['h']]) / 2
        b = np.dot(A, -b) + b
        im_dict['rotation'] = np.dot(A, im_dict['rotation'])
        im_dict['translation'] = np.dot(A, im_dict['translation']) + b

        return im_dict


class MultiScale(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, img):
        W, H = img.size
        sizes = [(int(W*ratio), int(H*ratio)) for ratio in self.scales]
        imgs = []
        [imgs.append(img.resize(size, Image.BILINEAR)) for size in sizes]
        return imgs


class ImageAugCompose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb


if __name__ == '__main__':
    flip = HorizontalFlip(p = 1)
    crop = RandomCrop((321, 321))
    rscales = RandomScale((0.75, 1.0, 1.5, 1.75, 2.0))
    img = Image.open('data/img.jpg')
    lb = Image.open('data/label.png')