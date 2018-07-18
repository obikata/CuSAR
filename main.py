# SAR processing with chirp scaling

##########################################
############## PROCESS FLOW ##############
# Raw data -->
# Azimuth FFT
# Chirp scaling / Range scaling
# Range FFT
# Bulk RCMC & Range compression
# Range IFFT
# PHase correction
# Azimuth compression
# Azimuth IFFT
# --> Compressed image 
##########################################

import pycuda.gpuarray as gpuarray
import numpy as np
import re
import time
import skcuda.fft as cu_fft
import matplotlib.pyplot as plt
import argparse
import glob
import os
import read_bin_as
import math
import sys
import struct

from phase_func import *
from read_bin_as import *
from skcuda import linalg
from skimage.exposure import rescale_intensity

import pycuda.autoinit
linalg.init()

def read_rec_as_arr(fp,nc,nl,skip):

    x = np.asarray(np.zeros((nl,nc))+1j*np.zeros((nl,nc)),np.complex64)

    fp.seek(720+8,0)
    nrec =  int.from_bytes(fp.read(4), byteorder='big')
    fp.seek(720+20,0)
    nzp0 =  int.from_bytes(fp.read(4), byteorder='big') # left padding
    nsamp =  int.from_bytes(fp.read(4), byteorder='big') # SAR sample length
    nzp1 =  int.from_bytes(fp.read(4), byteorder='big') # right padding
    npre = 412 # prefix length

    # debug
    # print(nrec)
    # print((nsamp+nzp0+nzp1)*2+npre)

    fp.seek(720+nrec*skip,0)
    data = struct.unpack('>%s'%(int((nrec*nl)))+'B',fp.read(int(nrec*nl))) # read multi record as 8bit floating point
    data = np.array(data).reshape(nl,nrec) #reshape 1D to 2D
    
    data = (data[:,int(npre):int(nrec)]-16.0) # truncate the prefix
    x[slice(nl),slice(int(nsamp+nzp0+nzp1))] = np.asarray(data[:,::2] + 1j*data[:,1::2],np.complex64)
    x = np.transpose(x)

    # debug
    # plt.figure()
    # plt.plot(np.imag(x[:,1001]))
    # plt.figure()
    # xf = np.fft.fft(x,axis=0) # range fft
    # plt.plot(np.abs(np.fft.fftshift(xf[:,1001])))
    # plt.show()

    return x

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', default="/media/logicool/73F526A645B63D5B/NLCS/L10",help='')
    parser.add_argument('--gpu', '-g', default=0,help='') # 0=cpu, 1=gpu, 2=benchmarking (cpu vs gpu)
    parser.add_argument('--nproc', '-n', default=1,help='') # number of process
    args = parser.parse_args()
    sArrFilePath = glob.glob(args.path + '/*PALSAR_L10_Tomakomai*')
    flg = int(args.gpu)
    nproc = int(args.nproc)

    # open img and led files
    img_file_path = glob.glob(sArrFilePath[0] + '/*IMG*')
    led_file_path = glob.glob(sArrFilePath[0] + '/*LED*')
    img_file_name, img_file_ext = os.path.splitext(os.path.basename(img_file_path[0]))
    fp_img = open(img_file_path[0],'rb')
    fp_led = open(led_file_path[0],'rb')

    # get orbit sense
    if img_file_ext == '.__A':
        orbit_sense = 'ASCENDING'
    else:
        orbit_sense = 'DESCENDING'

    # read raw signal data 
    nc = 16384 # number of range cells
    nl = 8192 # number of range lines
    
    # phase functions
    pf1, pf2, pf3, laz, r_shift = phase_func(fp_img,fp_led,nc,nl,orbit_sense)
    nread = nl-laz # length of samples to be read for each process (laz = length of azimuth sample for each process)

    if flg == 1 or 2:

        # send phase functions to gpu
        pf1_gpu = gpuarray.to_gpu(pf1)
        pf2_gpu = gpuarray.to_gpu(pf2)
        pf3_gpu = gpuarray.to_gpu(pf3)

        # fft plan
        az_plan = cu_fft.Plan((nl,1), np.complex64, np.complex64, nc)
        rg_plan = cu_fft.Plan((nc,1), np.complex64, np.complex64, nl)
    
    # start L1 process

    if flg == 0:

        # cpu

        slc = np.asarray(np.zeros((nc,nproc*nread))+1j*np.zeros((nc,nproc*nread)),np.complex64)
        x0 = np.asarray(np.zeros((nc,nl))+1j*np.zeros((nc,nl)),np.complex64)
        x0[:,slice(0,laz)] = read_rec_as_arr(fp_img,nc,laz,0)

        for j in range(laz):
            x0[:,j]=np.roll(x0[:,j],int(r_shift[j]),axis=0)

        for i in range(nproc):
            x0[:,slice(laz,nl)] = read_rec_as_arr(fp_img,nc,nread,laz+i*nread)

            for k in range(laz,nl):
                x0[:,k]=np.roll(x0[:,k],int(r_shift[k+i*nread]),axis=0)

            x = np.fft.fft(x0,axis=1)
            x = x*pf1
            x = np.fft.fft(x,axis=0)
            x = x*pf2
            x = np.fft.ifft(x,axis=0)
            x = x*pf3
            x = np.fft.ifft(x,axis=1)
            slc[:,slice(i*nread,(i+1)*nread)] = x[:,slice(laz,nl)]
            x0 = np.roll(x0,-nread,axis=1)
            print(i)

        # plot cpu result
        
        plt.figure()
        img_amp = np.abs(slc[slice(9800),:])
        img_amp = rescale_intensity(img_amp,in_range=(0, np.max(img_amp)/20),out_range=(0,255))
        plt.imshow(np.uint8(img_amp),cmap='gray')
        plt.show()

    elif flg == 1:
    
        # gpu

        slc_gpu = np.asarray(np.zeros((nc,nproc*nread))+1j*np.zeros((nc,nproc*nread)),np.complex64)
        x0 = np.empty((nc, nl, 1), np.complex64)
        x0[:,slice(0,laz),0] = read_rec_as_arr(fp_img,nc,laz,0)

        for j in range(laz):
            x0[:,j,0]=np.roll(x0[:,j,0],int(r_shift[j]),axis=0)

        for i in range(nproc):
            x0[:,slice(laz,nl),0] = read_rec_as_arr(fp_img,nc,nread,laz+i*nread)

            for k in range(laz,nl):
                x0[:,k,0]=np.roll(x0[:,k,0],int(r_shift[k+i*nread]),axis=0)

            x_gpu = gpuarray.to_gpu(x0)
            xt_gpu = gpuarray.to_gpu(np.empty((nl, nc, 1), np.complex64))
            cu_fft.fft(x_gpu, x_gpu, az_plan)
            x_gpu[:,:,0] = linalg.misc.multiply(x_gpu[:,:,0],pf1_gpu)
            xt_gpu[:,:,0] = linalg.transpose(x_gpu[:,:,0])
            cu_fft.fft(xt_gpu, xt_gpu, rg_plan)
            x_gpu[:,:,0] = linalg.transpose(xt_gpu[:,:,0])
            x_gpu[:,:,0] = linalg.misc.multiply(x_gpu[:,:,0],pf2_gpu)
            xt_gpu[:,:,0] = linalg.transpose(x_gpu[:,:,0])
            cu_fft.ifft(xt_gpu, xt_gpu, rg_plan, True)
            x_gpu[:,:,0] = linalg.transpose(xt_gpu[:,:,0])
            x_gpu[:,:,0] = linalg.misc.multiply(x_gpu[:,:,0],pf3_gpu)
            cu_fft.ifft(x_gpu, x_gpu, az_plan, True)
            slc_gpu[:,slice(i*nread,(i+1)*nread)] = x_gpu[:,slice(laz,nl),0].get()
            x0 = np.roll(x0,-nread,axis=1)
            print(i)
                
        # plot gpu result
        plt.figure()
        img_amp = np.abs(slc_gpu[slice(9800),:])
        img_amp = rescale_intensity(img_amp,in_range=(0, np.max(img_amp)/20),out_range=(0,255))
        plt.imshow(np.uint8(img_amp),cmap='gray')
        plt.show()

    elif flg == 2:
    
        # cpu

        start = time.time()

        slc = np.asarray(np.zeros((nc,nproc*nread))+1j*np.zeros((nc,nproc*nread)),np.complex64)
        x0 = np.asarray(np.zeros((nc,nl))+1j*np.zeros((nc,nl)),np.complex64)
        x0[:,slice(0,laz)] = read_rec_as_arr(fp_img,nc,laz,0)

        for j in range(laz):
            x0[:,j]=np.roll(x0[:,j],int(r_shift[j]),axis=0)

        for i in range(nproc):
            x0[:,slice(laz,nl)] = read_rec_as_arr(fp_img,nc,nread,laz+i*nread)

            for k in range(laz,nl):
                x0[:,k]=np.roll(x0[:,k],int(r_shift[k+i*nread]),axis=0)

            x = np.fft.fft(x0,axis=1)
            x = x*pf1
            x = np.fft.fft(x,axis=0)
            x = x*pf2
            x = np.fft.ifft(x,axis=0)
            x = x*pf3
            x = np.fft.ifft(x,axis=1)
            slc[:,slice(i*nread,(i+1)*nread)] = x[:,slice(laz,nl)]
            x0 = np.roll(x0,-nread,axis=1)
            print(i)

        elapsed_time = time.time() - start
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

        # gpu

        start = time.time()

        slc_gpu = np.asarray(np.zeros((nc,nproc*nread))+1j*np.zeros((nc,nproc*nread)),np.complex64)
        x0 = np.empty((nc, nl, 1), np.complex64)
        x0[:,slice(0,laz),0] = read_rec_as_arr(fp_img,nc,laz,0)

        for j in range(laz):
            x0[:,j,0]=np.roll(x0[:,j,0],int(r_shift[j]),axis=0)

        for i in range(nproc):
            x0[:,slice(laz,nl),0] = read_rec_as_arr(fp_img,nc,nread,laz+i*nread)

            for k in range(laz,nl):
                x0[:,k,0]=np.roll(x0[:,k,0],int(r_shift[k+i*nread]),axis=0)

            x_gpu = gpuarray.to_gpu(x0)
            xt_gpu = gpuarray.to_gpu(np.empty((nl, nc, 1), np.complex64))
            cu_fft.fft(x_gpu, x_gpu, az_plan)
            x_gpu[:,:,0] = linalg.misc.multiply(x_gpu[:,:,0],pf1_gpu)
            xt_gpu[:,:,0] = linalg.transpose(x_gpu[:,:,0])
            cu_fft.fft(xt_gpu, xt_gpu, rg_plan)
            x_gpu[:,:,0] = linalg.transpose(xt_gpu[:,:,0])
            x_gpu[:,:,0] = linalg.misc.multiply(x_gpu[:,:,0],pf2_gpu)
            xt_gpu[:,:,0] = linalg.transpose(x_gpu[:,:,0])
            cu_fft.ifft(xt_gpu, xt_gpu, rg_plan, True)
            x_gpu[:,:,0] = linalg.transpose(xt_gpu[:,:,0])
            x_gpu[:,:,0] = linalg.misc.multiply(x_gpu[:,:,0],pf3_gpu)
            cu_fft.ifft(x_gpu, x_gpu, az_plan, True)
            slc_gpu[:,slice(i*nread,(i+1)*nread)] = x_gpu[:,slice(laz,nl),0].get()
            x0 = np.roll(x0,-nread,axis=1)
            print(i)

        elapsed_time = time.time() - start
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

        # check success status
        print('Success status: %r' % np.allclose(np.abs(slc), np.abs(slc_gpu), atol=1e-1))

if __name__ == '__main__':
    main()
