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



batch_size = 4

data_size = 64
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

def fast_validate(model, ldrs_list, hdr_list, expos_list, crop = True, write=False, multi_output=False, use_pre=False, use_vgg=False):
    def formulate_hdr(x):
        assert len(x.shape) == 4
        _hdr = x
        _hdr = np.round(_hdr[0]*65535).astype(np.uint16)
        return _hdr
    valid_psnr = 0
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
        t_time = time.time()
        if use_pre:
            pre = hdr_merge(ldrs[:,:,0:3], ldrs[:,:,3:6], ldrs[:,:,6:9], expos)
            pre = np.expand_dims(pre, 0)
            _hdr = model.predict([x1, x2, x3, pre])
            _hdr = np.clip(_hdr, 0, 1)
        else:
            _hdr = model.predict([x1, x2, x3])
        
        if use_vgg:
            _hdr = _hdr[0]

        if multi_output:
            _hdr = _hdr[-1]
        _psnr = 10*log10(1/np.mean((u_law_numpy(_hdr) - u_law_numpy(hdr))**2))
        valid_psnr += _psnr
        if write:
            print('%s: psnr = %.2f, time=%.2f'%(get_name(i), _psnr, time.time()-t_time))
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
 	os.environ["CUDA_VISIBLE_DEVICES"]="2"

training_data_path = 'h5_data_raw/Training/'
test_data_path = 'h5_data_raw/Test/'

#build model
if multi_output:
    b_model = bicubic_model()
if use_vgg:
    vgg_model = pretrain_vgg16()

model = DPN(64, input_chs, nBlocks=3, Res=False, use_vgg=use_vgg, vgg_model=vgg_model)
_compile(model, lr[lr_index], loss_func, use_vgg)
test_ldrs_list, test_hdr_list, test_expos_list = load_test_sequences(test_data_path)
train_file_list = os.listdir(training_data_path)

ldrs_list = []
hdr_list = []
if use_pre:
    pre_list = []

min_loss = 0
compile_flag = False
fail_count = 0

min_loss = fast_validate(model, test_ldrs_list, test_hdr_list, test_expos_list, crop=True,write=False ,multi_output=multi_output, use_pre=use_pre, use_vgg=use_vgg)
print(min_loss)
#exit(0)

for epoch in range(20000):
    random.shuffle(train_file_list)
    total_batches = len(train_file_list)//batch_size
    itr = 0
    for f in train_file_list:
        h5_file = h5.File(training_data_path+f, 'r')
        _x = random.randint(0, CROP_SIZE-data_size)
        _y = random.randint(0, CROP_SIZE-data_size)
        _ldr = h5_file['ldr'][_y:_y+data_size, _x:_x+data_size]
        _hdr = h5_file['hdr'][_y:_y+data_size, _x:_x+data_size]
        expos = h5_file['expos']
        method = random.randint(0, 5)
        _ldr = data_augmentation(_ldr, method)
        _hdr = data_augmentation(_hdr, method)
        _ldr = pre_process_train16(_ldr)
        _x1 = _ldr[:,:,0:3]
        _x2 = _ldr[:,:,3:6]
        _x3 = _ldr[:,:,6:9]
        if use_pre:
            _pre = hdr_merge(_x1, _x2, _x3, expos)
            pre_list.append(np.expand_dims(_pre, 0))

        _ldr = np.concatenate([_x1, LDR2HDR(_x1, expos[0]/expos[1]), 
                               _x2, LDR2HDR(_x2, expos[1]/expos[1]),
                               _x3, LDR2HDR(_x3, expos[2]/expos[1])], axis=-1)
        ldrs_list.append(np.expand_dims(_ldr, 0))
        hdr_list.append(np.expand_dims(_hdr, 0))
        
        if len(ldrs_list) == batch_size:
            itr += 1
            ldrs_batch = np.concatenate(ldrs_list, axis=0)
            hdr_batch = np.concatenate(hdr_list, axis=0)
            if use_vgg:
                hdr_batch = [hdr_batch, vgg_process(hdr_batch)]
            ldrs_list = []
            hdr_list = []
            if use_pre:
                pre_batch = np.concatenate(pre_list, axis=0)
                pre_list = []
                loss = model.train_on_batch([ldrs_batch[:,:,:,0:input_chs], 
                                    ldrs_batch[:,:,:,input_chs:input_chs*2], 
                                    ldrs_batch[:,:,:,input_chs*2:input_chs*3],
                                    pre_batch], 
                                    hdr_batch)
            else:
                loss = model.train_on_batch([ldrs_batch[:,:,:,0:input_chs], 
                                    ldrs_batch[:,:,:,input_chs:input_chs*2], 
                                    ldrs_batch[:,:,:,input_chs*2:input_chs*3]], 
                                    hdr_batch)
            print('epoch %d %d/%d ---> loss: %.6f'%(epoch+1, itr, total_batches, loss), end='\r')

            print('Training is done. The best psnr = %.4f'%(min_loss))
            exit(0)
