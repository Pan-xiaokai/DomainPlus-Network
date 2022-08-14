import os, cv2, random, datetime, time
import tensorflow as tf
import numpy as np
from utils import *
import h5py as h5
from tensorflow.keras import optimizers
from ahdr_model import *
from my_model import *
from math import log10
from data import *
from common import *
from skimage.measure import compare_psnr, compare_ssim


batch_size = 4
lr = [1e-4, 5e-5, 2.5e-5, 1e-5, 5e-6, 1e-6]
lr_index = 0
data_size = 256
input_chs = 6
multi_output = False
multi_gpu = False
use_pre = False
use_vgg = False
vgg_model = None

loss_func = loss_func_v3

def validate(model, test_data_path, write=False):
    test_file_list = os.listdir(test_data_path)
    valid_psnr = 0
    for f in test_file_list:
        h5_file = h5.File(test_data_path+f, 'r')
        ldrs = np.expand_dims(h5_file['ldr'], 0)
        hdr = np.expand_dims(h5_file['hdr'], 0)
        h = ldrs.shape[1]
        w = ldrs.shape[2]
        h = h - h%16
        w = w - w%16
        t_time = time.clock()
        _hdr = model.predict([ldrs[:,0:h,0:w,0:input_chs],
                              ldrs[:,0:h,0:w,input_chs:input_chs*2],
                              ldrs[:,0:h,0:w,input_chs*2:input_chs*3]])
        if write:
            print(time.clock()-t_time)
        valid_psnr += 10*log10(1/np.mean((u_law_numpy(_hdr) - u_law_numpy(hdr[:,0:h,0:w,:]))**2))
        if write:
            _hdr = np.clip(u_law_numpy(_hdr), 0, 1)
            _hdr = np.round(_hdr[0]*255).astype(np.uint8)
            cv2.imwrite(f[:-3]+'_ssim.png',_hdr)
    valid_psnr = valid_psnr / (len(test_file_list))
    return valid_psnr

def load_test_sequences(test_data_path):
    test_file_list = os.listdir(test_data_path)
    ldrs_list = []
    hdr_list = []
    expos_list = []
    for f in test_file_list:
        h5_file = h5.File(test_data_path+f, 'r')
        ldrs_list.append(h5_file['ldr'])
        hdr_list.append(h5_file['hdr'])
        expos_list.append(h5_file['expos'])

    return ldrs_list, hdr_list, expos_list

def SSIMnp(y_true , y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01*7)
    c2 = np.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom

def fast_validate(model, ldrs_list, hdr_list, expos_list, crop = True, write=False, multi_output=False, use_pre=False, use_vgg=False):
    def formulate_hdr(x):
        assert len(x.shape) == 4
        _hdr = np.clip(u_law_numpy(x), 0, 1)
        _hdr = np.round(_hdr[0]*65535).astype(np.uint16)
        return _hdr
    valid_psnr = 0
    valid_ssim = 0
    valid_psnr_mu = 0
    valid_psnr_L = 0
    valid_ssim_mu = 0
    valid_ssim_L = 0
    for i, ldrs in enumerate(ldrs_list, 0):
        hdr = hdr_list[i]
        expos = expos_list[i]
        if crop:
            h, w = ldrs.shape[0], ldrs.shape[1]
            h = h - h%16
            w = w - w%16
            hdr = hdr[0:h, 0:w, :]
            ldrs = ldrs[0:h, 0:w, :]
        ldrs = pre_process_test16(ldrs)
        _x1 = LDR2HDR(ldrs[:,:,0:3], expos[0]/expos[1])
        _x2 = LDR2HDR(ldrs[:,:,3:6], expos[1]/expos[1])
        _x3 = LDR2HDR(ldrs[:,:,6:9], expos[2]/expos[1])
        x1 = np.expand_dims(np.concatenate([ldrs[:,:,0:3], _x1], axis=-1), 0)
        x2 = np.expand_dims(np.concatenate([ldrs[:,:,3:6], _x2], axis=-1), 0)
        x3 = np.expand_dims(np.concatenate([ldrs[:,:,6:9], _x3], axis=-1), 0)
        hdr = np.expand_dims(hdr, 0)
        t_time = time.clock()
        if use_pre:
            pre = hdr_merge(ldrs[:,:,0:3], ldrs[:,:,3:6], ldrs[:,:,6:9], expos)
            pre = np.expand_dims(pre, 0)
            _hdr = model.predict([x1, x2, x3, pre])
            _hdr = np.clip(_hdr, 0, 1)
        else:
            _hdr,att_f_x1x2, att_f_x3x2 = model.predict([x1, x2, x3])
            end_time = time.clock()
        
        if use_vgg:
            _hdr = _hdr[0]

        if multi_output:
            _hdr = _hdr[-1]

        _psnr = 10*log10(1/np.mean((u_law_numpy(_hdr) - u_law_numpy(hdr))**2))
        valid_psnr += _psnr

        if write:
            print('%s: psnr = %.2f, time=%.2f'%(get_name(i), _psnr, end_time-t_time))
            cv2.imwrite('output/' +get_name(i+1)+'DPN.tif',formulate_hdr(_hdr))
    valid_psnr = valid_psnr / (len(ldrs_list))

    return valid_psnr
def pre_merge(ldrs_list, expos_list):
    for i, ldrs in enumerate(ldrs_list, 0):
        expos = expos_list[i]
        ldrs = pre_process(ldrs[:])
        hdr = hdr_merge(ldrs[:,:,0:3], ldrs[:,:,3:6], ldrs[:,:,6:9], expos)
        hdr = u_law_numpy(hdr)
        hdr = np.round(hdr*255).astype(np.uint8)
        cv2.imwrite('../pre_merge/'+get_name(i+1)+'.png',hdr)

def vgg_process(x):
    x = (x - 0.5)*2
    return vgg_model.predict(x)

def _compile(model, lr, loss_func, use_vgg=False):
    if use_vgg:
        model.compile(optimizer = optimizers.Adam(lr = lr), loss = [loss_func, 'mae'], loss_weights=[1.0, 0.1])
    else:
        model.compile(optimizer = optimizers.Adam(lr = lr), loss = loss_func)

if multi_gpu:	
	print ('use multi gpu mode!')
	os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
else:
	os.environ["CUDA_VISIBLE_DEVICES"]="3"

test_data_path = 'h5_data_raw/'
#build model
if multi_output:
    b_model = bicubic_model()
if use_vgg:
    vgg_model = pretrain_vgg16()

model = modelv2(64, input_chs, nBlocks=3, Res=False, use_vgg=use_vgg, vgg_model=vgg_model)

test_ldrs_list, test_hdr_list, test_expos_list = load_test_sequences(test_data_path)

model.load_weights('model_new/mymodelv2_best.h5')

print(fast_validate(model, test_ldrs_list, test_hdr_list, test_expos_list, crop=True, write=True, use_pre=use_pre))
exit(0)
