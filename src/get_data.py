# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 20:49:06 2021

@author: anshu
"""

import numpy as np

def get_training_data(batch_size=32):
    T=np.load('./data/training_data.npy')
    L=np.load('./data/training_labels.npy')
    
    nums=int(len(T)/batch_size)
    
    T=np.array_split(T,nums)
    L=np.array_split(L,nums)
    
    return T,L


def get_validation_data():
    T=np.load('./data/val_data.npy')
    L=np.load('./data/val_labels.npy')    
    return T,L