# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 16:25:26 2022

@author: lixyxd
"""

import numpy as np
import torch
import scipy.io as io
import random


def create_zidian(G_dir, GG_dir):
    A = np.load(G_dir)
    AA = np.load(GG_dir)
    # AT = torch.from_numpy(A.conj().T).to(torch.complex128) #torch.complex128
    # A = torch.from_numpy(A).to(torch.complex128)  #torch.complex128
    # AA = torch.from_numpy(AA).to(torch.complex128)  #torch.complex128
    AT = torch.from_numpy(A.conj().T).to(torch.complex64)
    A = torch.from_numpy(A).to(torch.complex64)
    AA = torch.from_numpy(AA).to(torch.complex64)

    zidian = {'A': A, 'AT': AT, 'AA': AA}
    zidiancuda = {'A': A.cuda(), 'AT': AT.cuda(), 'AA': AA.cuda()}
    
    #    return zidian, zidiancuda
    return zidian, zidiancuda
