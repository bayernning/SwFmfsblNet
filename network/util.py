# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 10:35:18 2023

@author: Administrator
"""

import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
    # compare_psnr, compare_ssim, compare_nrmse, compare_mse


def ssim_index(im1, im2):
    '''
    Input:
        im1, im2: np.uint8 format
    '''

    out = compare_ssim(im1.squeeze(), im2.squeeze(), gaussian_weights=True, data_range=2)

    return out


# def batch_NMSE(img, imclean):
#     Img = img.data.cpu().numpy()
#     Iclean = imclean.data.cpu().numpy()
#     #    Img = img_as_ubyte(Img)
#     #    Iclean = img_as_ubyte(Iclean)
#     NMSE = 0
#     for i in range(Img.shape[0]):
#         NMSE += compare_nrmse(Iclean[i, :, :, :].squeeze(), Img[i, :, :, :].squeeze())
#     #        RMSE +=
#     return NMSE / Img.shape[0]


def batch_RMSE(img, imclean):
    _, _, H, W = imclean.shape
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    #    Img = img_as_ubyte(Img)
    #    Iclean = img_as_ubyte(Iclean)
    RMSE = 0
    for i in range(Img.shape[0]):
        RMSE += np.sqrt(compare_mse(Iclean[i, :, :, :].squeeze(), Img[i, :, :, :].squeeze()))
    return RMSE / Img.shape[0]


def batch_PSNR(img, imclean):
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    #    Img = img_as_ubyte(Img)
    #    Iclean = img_as_ubyte(Iclean)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :].squeeze(), Img[i, :, :, :].squeeze())
    return (PSNR / Img.shape[0])


def batch_SSIM(img, imclean):
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    #    Img = img_as_ubyte(Img)
    #    Iclean = img_as_ubyte(Iclean)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += ssim_index(Iclean[i, :, :, :].transpose((1, 2, 0)), Img[i, :, :, :].transpose((1, 2, 0)))
    return (SSIM / Img.shape[0])


'''
def NMSE(x_hat, x):
    x_hat = x_hat[:,0,:,:]+1j*x_hat[:,1,:,:]
    x = x[:,0,:,:]+1j*x[:,1,:,:]
    x_hat = x_hat.data.cpu().numpy()
    x = x.data.cpu().numpy()
    NMSE = 0
    for i in range(x.shape[0]):
        a = np.linalg.norm((x_hat-x)[i,:,:],ord=2)/np.linalg.norm(x[i,:,:],ord=2)
        NMSE += a**2
    return (NMSE/x.shape[0])


def PSNR(x_hat, x):
    x_hat = x_hat[:,0,:,:]+1j*x_hat[:,1,:,:]
    x = x[:,0,:,:]+1j*x[:,1,:,:]
    x_hat = x_hat.data.cpu().numpy()
    x = x.data.cpu().numpy()
    PSNR = 0
    for i in range(x.shape[0]):
        a = 255**2*x.shape[-2]*x.shape[-1]/(np.linalg.norm((x_hat-x)[i,:,:],ord=2)**2)
        PSNR += 10*np.log10(a)
    return (PSNR/x.shape[0])
'''
