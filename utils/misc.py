#!/usr/bin/env python

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as VF
import torchvision.utils as vutils
from PIL import Image
import os
from torchvision import models
import json
from torch.nn import functional as F

def load_image(image_path):
    """Loads image as a PIL RGB image.

        Args:
            - **image_path (str) - **: A path to the image

        Returns:
            An instance of PIL.Image.Image in RGB

    """

    return Image.open(image_path).convert('RGB')


def get_transform(resize_size=224, center_crop_size=256):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    # if center_crop_size:
    #     transform = transforms.Compose([
    #         transforms.Resize(resize_size), # the smaller edge of the image will be
    #         # matched to this number maintaining the aspect ratio
    #         transforms.CenterCrop(center_crop_size),
    #         transforms.ToTensor(),
    #         transforms.Normalize(means, stds)
    #     ])
    # elif resize_size:
    #     transform = transforms.Compose([
    #         transforms.Resize(resize_size),
    #         transforms.ToTensor(),
    #         transforms.Normalize(means, stds)
    #     ])
    # else:
    #     transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(means, stds)
    #     ])


    transforms_list = [transforms.Resize(resize_size) if resize_size else None, # if the resize size is an int,
    # the smaller edge of the image will be matched to this number maintaining the aspect ratio, while if it is tuple,
    # it will be resized to that size by ignoring aspect ratio.
    transforms.CenterCrop(center_crop_size) if center_crop_size else None,
    transforms.ToTensor(),
    transforms.Normalize(means, stds)]

    transforms_list =  [i for i in transforms_list if i]
    transform = transforms.Compose(transforms_list)
    return transform

def apply_transforms(image, resize_size=256, center_crop_size=224, requires_grad=True, expand_dim=False):
    """Transforms a PIL image to torch.Tensor.

    Applies a series of tranformations on PIL image including a conversion
    to a tensor. The returned tensor has a shape of :math:`(N, C, H, W)` and
    is ready to be used as an input to neural networks.

    First the image is resized to 256, then cropped to 224. The `means` and
    `stds` for normalisation are taken from numbers used in ImageNet, as
    currently developing the package for visualizing pre-trained models.

    The plan is to to expand this to handle custom size/mean/std.

    Args:
        image (PIL.Image.Image or numpy array)
        size (int, optional, default=224): Desired size (width/height) of the
            output tensor

    Shape:
        Input: :math:`(C, H, W)` for numpy array
        Output: :math:`(N, C, H, W)`

    Returns:
        torch.Tensor (torch.float32): Transformed image tensor

    Note:
        Symbols used to describe dimensions:
            - N: number of Images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image

    """

    if not isinstance(image, Image.Image):
        image = VF.to_pil_image(image)

    transform = get_transform(resize_size=resize_size, center_crop_size=center_crop_size)
    tensor = transform(image)
    if expand_dim:
        tensor = tensor.unsqueeze(0)

    if requires_grad:
        tensor.requires_grad = True

    return tensor


def apply_transforms_v0(image, size=224):
    """Transforms a PIL image to torch.Tensor.

    Applies a series of tranformations on PIL image including a conversion
    to a tensor. The returned tensor has a shape of :math:`(N, C, H, W)` and
    is ready to be used as an input to neural networks.

    First the image is resized to 256, then cropped to 224. The `means` and
    `stds` for normalisation are taken from numbers used in ImageNet, as
    currently developing the package for visualizing pre-trained models.

    The plan is to to expand this to handle custom size/mean/std.

    Args:
        image (PIL.Image.Image or numpy array)
        size (int, optional, default=224): Desired size (width/height) of the
            output tensor

    Shape:
        Input: :math:`(C, H, W)` for numpy array
        Output: :math:`(N, C, H, W)`

    Returns:
        torch.Tensor (torch.float32): Transformed image tensor

    Note:
        Symbols used to describe dimensions:
            - N: number of Images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image

    """

    if not isinstance(image, Image.Image):
        image = VF.to_pil_image(image)

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])

    tensor = transform(image).unsqueeze(0)

    tensor.requires_grad = True

    return tensor

def denormalize(tensor):
    """Reverses the normalisation on a tensor.

    Performs a reverse operation on a tensor, so the pixel value range is
    between 0 and 1. Useful for when plotting a tensor into an image.

    Normalisation: (image - mean) / std
    Denormalisation: image * std + mean

    Args:
        tensor (torch.Tensor, dtype=torch.float32): Normalized image tensor

    Shape:
        Input: :math:`(N, C, H, W)`
        Output: :math:`(N, C, H, W)` (same shape as input)

    Return:
        torch.Tensor (torch.float32): Demornalised image tensor with pixel
            values between [0, 1]

    Note:
        Symbols used to describe dimensions:
            - N: number of Images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image

    """

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    denormalized = tensor.clone()

    # for channel, mean, std in zip(denormalized[0], means, stds):
    #     channel.mul_(std).add_(mean)

    for i in range(3):
        channel = denormalized[0][i]
        channel = channel * stds[i]
        channel = channel + means[i]
        denormalized[0][i] = channel

    return denormalized


def tensor_to_img(tensor):
    return Image.fromarray(np.transpose(np.uint8(denormalize(tensor)[0]*255), (1 ,2 ,0)))

def xmkdir(path):
    r"""Create a directory path recursively.

    The function creates :attr:`path` if the directory does not exist.

    Args::
        path (str): path to create.
    """
    if path is not None and not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            # Race condition in multi-processing.
            pass


def standardize_and_clip(tensor, min_value=0.0, max_value=1.0):
    """Standardizes and clips input tensor.

    Standardize the input tensor (mean = 0.0, std = 1.0), ensures std is 0.1
    and clips it to values between min/max (default: 0.0/1.0).

    Args:
        tensor (torch.Tensor):
        min_value (float, optional, default=0.0)
        max_value (float, optional, default=1.0)

    Shape:
        Input: :math:`(C, H, W)`
        Output: Same as the input

    Return:
        torch.Tensor (torch.float32): Normalised tensor with values between
            [min_value, max_value]

    """

    tensor = tensor.detach().cpu()

    mean = tensor.mean()
    std = tensor.std()
    if std == 0:
        std += 1e-7

    standardized = tensor.sub(mean).div(std).mul(0.1)
    clipped = standardized.add(0.5).clamp(min_value, max_value)

    return clipped


def format_for_plotting(tensor):
    """Formats the shape of tensor for plotting.

    Tensors typically have a shape of :math:`(N, C, H, W)` or :math:`(C, H, W)`
    which is not suitable for plotting as Images. This function formats an
    input tensor :math:`(H, W, C)` for RGB and :math:`(H, W)` for mono-channel
    data.

    Args:
        tensor (torch.Tensor, torch.float32): Image tensor

    Shape:
        Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`
        Output: :math:`(H, W, C)` or :math:`(H, W)`, respectively

    Return:
        torch.Tensor (torch.float32): Formatted image tensor (detached)

    Note:
        Symbols used to describe dimensions:
            - N: number of Images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image

    """

    has_batch_dimension = len(tensor.shape) == 4
    formatted = tensor.clone()

    if has_batch_dimension:
        formatted = tensor.squeeze(0)

    if formatted.shape[0] == 1:
        return formatted.squeeze(0).detach()
    else:
        return formatted.permute(1, 2, 0).detach()


def visualize(input_, gradients, save_path=None, cmap='viridis', alpha=0.7):
    """ Method to plot the explanation.

        # Arguments
            input_: Tensor. Original image.
            gradients: Tensor. Saliency map result.
            save_path: String. Defaults to None.
            cmap: Defaults to be 'viridis'.
            alpha: Defaults to be 0.7.

    """

    input_ = format_for_plotting(denormalize(input_))
    gradients = format_for_plotting(standardize_and_clip(gradients))

    subplots = [
        ('Input image', [(input_, None, None)]),
        ('Saliency map across RGB channels', [(gradients, None, None)]),
        ('Overlay', [(input_, None, None), (gradients, cmap, alpha)])
    ]

    num_subplots = len(subplots)

    fig = plt.figure(figsize=(16, 3))

    for i, (title, images) in enumerate(subplots):
        ax = fig.add_subplot(1, num_subplots, i + 1)
        ax.set_axis_off()

        for image, cmap, alpha in images:
            ax.imshow(image, cmap=cmap, alpha=alpha)

        ax.set_title(title)
    if save_path is not None:
        plt.savefig(save_path)


def basic_visualize(input_, gradients, save_path=None, weight=None, cmap='viridis', alpha=0.7):
    """ Method to plot the explanation.

        # Arguments
            input_: Tensor. Original image.
            gradients: Tensor. Saliency map result.
            save_path: String. Defaults to None.
            cmap: Defaults to be 'viridis'.
            alpha: Defaults to be 0.7.

    """
    input_ = format_for_plotting(denormalize(input_))
    gradients = format_for_plotting(standardize_and_clip(gradients))

    subplots = [
        ('Saliency map across RGB channels', [(gradients, None, None)]),
        ('Overlay', [(input_, None, None), (gradients, cmap, alpha)])
    ]

    num_subplots = len(subplots)

    fig = plt.figure(figsize=(4, 4))

    for i, (title, images) in enumerate(subplots):
        ax = fig.add_subplot(1, num_subplots, i + 1)
        ax.set_axis_off()

        for image, cmap, alpha in images:
            ax.imshow(image, cmap=cmap, alpha=alpha)

    if save_path is not None:
        plt.savefig(save_path)


def show_heatmap(mask, title=None):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (Tensor): shape (1, 1, H, W)
    Return:
        heatmap (Tensor): shape (1, 3, H, W)
        :param title:
    """
    mask = (mask - mask.min()).div(mask.max() - mask.min()).data
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze().float()), cv2.COLORMAP_JET)  # [H, W, 3]
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)

    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    if title is not None:
        vutils.save_image(heatmap, title)

    return heatmap


def show_cam(img, mask, title=None, nomalise=True):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        img (Tensor): shape (1, 3, H, W)
        mask (Tensor): shape (1, 1, H, W)
    Return:
        heatmap (Tensor): shape (3, H, W)
        cam (Tensor): synthesized GradCAM cam of the same shape with heatmap.
        :param title:
    """
    mask = (mask - mask.min()).div(mask.max() - mask.min()).data
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze().float()), cv2.COLORMAP_JET)  # [H, W, 3]
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)

    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    if nomalise:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # cam = heatmap + img.cpu()
    cam = 1 * (1 - mask ** 0.8) * img + (mask ** 0.8) * heatmap
    # cam = (cam - cam.min()) / (cam.max() - cam.min())
    if title is not None:
        vutils.save_image(cam, title)

    return cam

def save_img_with_heatmap(img, mask, title=None, style=None, normalise=True):
    """Display heatmap on top of the image.
    Args:
        img (PIL): RGB
        mask (2d array): shape (H, W)
    Return:
        heatmap (Tensor): shape (3, H, W)
        cam (Tensor): synthesized GradCAM cam of the same shape with heatmap.
        :param title:
    """

    if normalise:
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)  # [H, W, 3]

    heatmap = heatmap[:, :, ::-1]
    if style == 'zhou':
        img_with_heatmap = heatmap * 0.3 + np.array(img) * 0.5
    else:
        # if normalise:
        #     heatmap = heatmap - np.min(heatmap)
        #     heatmap = heatmap/np.max(heatmap)
        # heatmap = np.uint8(255 * heatmap)
        mask = np.expand_dims(mask, axis=2)
        img_with_heatmap = 1 * (1 - mask ** 0.8) * img + (mask ** 0.8) * heatmap
        # img_with_heatmap = 1 * (1 - mask) * img + (mask) * heatmap
        # cam = (cam - cam.min()) / (cam.max() - cam.min())

    img_with_heatmap = img_with_heatmap[:, :, ::-1]
    if title is not None:
        cv2.imwrite(title, img_with_heatmap)
        # Image.fromarray(np.uint8(img_with_heatmap)).save(title)

    return img_with_heatmap

def save_heatmap(heatmap, title, normalise=None):
    if normalise:
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / np.max(heatmap)
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    if title is not None:
        cv2.imwrite(title, heatmap)
    return heatmap




# def save_zhou_style_img_with_heatmap(img, mask, title=None):
#     """Display heatmap on top of the image.
#     Args:
#         img (PIL): RGB
#         mask (2d array): shape (H, W)
#     Return:
#         heatmap (Tensor): shape (3, H, W)
#         cam (Tensor): synthesized GradCAM cam of the same shape with heatmap.
#         :param title:
#     """
#
#     mask = mask - np.min(mask)
#     mask = mask / np.max(mask)
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)  # [H, W, 3]
#
#     img_with_heatmap = heatmap * 0.3 + np.array(img) * 0.5
#     if title is not None:
#         Image.fromarray(np.uint8(img_with_heatmap)).save(title)
#
#     return img_with_heatmap


def preprocess_img(cv_img):
    """Turn a opencv image into tensor and normalize it"""
    # revert the channels from BGR to RGB
    img = cv_img.copy()[:, :, ::-1]
    # convert tor tensor
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1))))
    # Normalize
    transform_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    norm_img = transform_norm(img).unsqueeze(0)

    return img, norm_img


# def convert_to_gray(x, percentile=99):
#     """
#     Args:
#         x: torch tensor with shape of (1, 3, H, W)
#         percentile: int
#     Return:
#         result: shape of (1, 1, H, W)
#     """
#     x_2d = torch.abs(x).sum(dim=1).squeeze(0)
#     v_max = np.percentile(x_2d, percentile)
#     v_min = torch.min(x_2d)
#     torch.clamp_((x_2d - v_min) / (v_max - v_min), 0, 1)
#     return x_2d.unsqueeze(0).unsqueeze(0)

def convert_to_gray(im_as_arr):
    """
        Converts 3d image to grayscale
    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def write_video(inputpath, outputname, img_num, fps=10):
    """Generate videos
    Args:
        input_path: the path for input Images
        output_name: the output name for the video
        img_num: the number of the input Images
        fps: frames per second
    """

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter(outputname, fourcc, fps, (1000, 1000))
    for i in range(img_num):
        img_no = i + 1
        print(inputpath + 'video' + str(img_no) + '.jpg')
        cv_img = cv2.imread(inputpath + 'video' + str(img_no) + '.jpg', 1)
        videoWriter.write(cv_img)
    videoWriter.release()


# def resize_img(img, maxsize):
#     # basewidth = _width
#     width, height = img.size
#     if width >= height:
#         wpercent = (maxsize / float(width))
#         height = int((float(height) * float(wpercent)))
#         width = maxsize
#     else:
#         wpercent = (maxsize / float(height))
#         width = int((float(width) * float(wpercent)))
#         height = maxsize
#
#     # hsize = int((float(img.size[1]) * float(wpercent)))
#     img = img.resize((width, height), Image.ANTIALIAS)
#     return img

def resize_img(img, basewidth):
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    return img


# def sliding_window(height, width,  stepSize, windowSize):
#     # slide a window across the image
#     for y in range(0, height - windowSize[1], stepSize):
#         for x in range(0, width - windowSize[0] , stepSize):
#             # yield the current window
#             yield (x, y, x + windowSize[0], y + windowSize[1])

def sliding_window(height, width,  stepSize, windowSize):
    # slide a window across the image
    windows = []
    y_flag = False
    for y in range(0, height, stepSize):
        x_flag = False
        for x in range(0, width, stepSize):
            # yield the current window
            x0 = x
            if width <= x + windowSize[0]:
                x1 = width
                x0 = width - windowSize[0]
                x_flag = True
            else :
                x1 = x0 + windowSize[0]

            y0 = y
            if height <= y + windowSize[1]:
                y1 = height
                y0 = height - windowSize[1]
                y_flag = True
            else :
                y1 = y + windowSize[1]

            windows.append((x0, y0, x1, y1))
            if x_flag: break

        if y_flag: break
    return windows

def check_path_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_id(img):
    net = models.resnet50(pretrained=True)
    net.eval()
    preprocess = get_transform(resize_size=(224, 224), center_crop_size=None)
    img_tensor = preprocess(img)
    img_variable = img_tensor.unsqueeze(0)
    logit = net(img_variable)
    # load the imagenet category list
    LABELS_file = 'utils/resources/imagenet_class_index.json'
    with open(LABELS_file) as f:
        classes = json.load(f)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # output the prediction
    for i in range(0, 5):
        print('{:.3f} -> {} {}'.format(probs[i], idx[i], classes[str(idx[i])][1]))