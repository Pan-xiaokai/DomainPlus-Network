import numpy as np
import cv2, os, math
import h5py as h5
from common import u_law_numpy, LDR2HDR

MAX_PIX_VAL_train = 255
MAX_PIX_VAL_test = 255
MAX_PIX_VAL_test16 = 65535

CROP_SIZE = 320
TRAINING_SCENE_PATH = 'datasets/Training/'
TEST_SCENE_PATH = 'datasets/Test/'

def read_expo(file_path):
    with open(file_path) as fp:
        lines = fp.readlines()
        assert len(lines) == 3
    return pow(2,float(lines[0])), pow(2,float(lines[1])), pow(2,float(lines[2]))


def get_name(count, length=6):
    s = str(count)
    l = len(s)
    for i in range(length-l):
        s = '0'+s
    return s
def list_filter(file_list, tail):
	r = []
	for f in file_list:
		s = os.path.splitext(f)
		if s[1] == tail:
			r.append(f)
	return r

def pre_process(img, val):
    img = img.astype(np.float32)
    img = img/val
    return img

def pre_process_train(img):
    img = img.astype(np.float32)
    img = img/MAX_PIX_VAL_train
    return img
    
def pre_process_train16(img):
    img = img.astype(np.float32)
    img = img/MAX_PIX_VAL_test16
    return img

def pre_process_test(img):
    img = img.astype(np.float32)
    img = img/MAX_PIX_VAL_test
    return img

def pre_process_test16(img):
    img = img.astype(np.float32)
    img = img/MAX_PIX_VAL_test16
    return img

def generate_anchors(shape, size):
   def process_line(l, size):
      n = math.ceil(l/size)
      step = 64
      pos = 0
      pos_list = []
      while(pos +size < l+step):
         if (pos+size) <= l:
            pos_list.append(pos)
         else:
            pos_list.append(l-size)
            break
         pos += step
      return pos_list
   h = shape[0]
   w = shape[1]
   pos_list = []
   h_list = process_line(h, size)
   w_list = process_line(w, size)
   for i in range(len(h_list)):
      for j in range(len(w_list)):
         pos_list.append((h_list[i], w_list[j]))
   return pos_list

def load_train_scene(scene_path, h5_path, total_count):
    file_list = os.listdir(scene_path)
    expos = read_expo(scene_path + 'exposure.txt')
    hdr = cv2.imread(scene_path + 'HDRImg.hdr', flags=-1)
    #load ldr inputs
    input_img_list = list_filter(file_list, '.tif')
    ldr_list = []
    assert len(input_img_list) == len(expos)
    for i, img_path in enumerate(input_img_list):
        img = cv2.imread(scene_path+img_path, flags =1)
        ldr_list.append(img)

    input_ldrs = np.concatenate(ldr_list, axis=-1)
    #crop image into patches for training
    anchors = generate_anchors(input_ldrs.shape, CROP_SIZE)
    for anchor in anchors:
        y = anchor[0]
        x = anchor[1]
        _ldr = input_ldrs[y:y+CROP_SIZE, x:x+CROP_SIZE]
        _hdr = hdr[y:y+CROP_SIZE, x:x+CROP_SIZE]
        if _ldr.shape[0]!=CROP_SIZE or _ldr.shape[1]!=CROP_SIZE:
            continue
        total_count = total_count + 1
        h5_file_name = h5_path + get_name(total_count)+'.h5'
        f = h5.File(h5_file_name, 'w')
        f['ldr'] = _ldr
        f['hdr'] = _hdr
        f['expos'] = np.array(expos)
            
    return total_count

def load_test_scene(scene_path, h5_path, total_count):
    file_list = os.listdir(scene_path)
    #load exposure times
    expos = read_expo(scene_path + 'exposure.txt')
    #load hdr groundtruth
    hdr = cv2.imread(scene_path + 'HDRImg.hdr', flags = cv2.IMREAD_ANYDEPTH)
    #load ldr inputs
    input_img_list = list_filter(file_list, '.tif')
    ldr_list = []
    assert len(input_img_list) == len(expos)
    for i, img_path in enumerate(input_img_list):
        img = cv2.imread(scene_path+img_path, -1)
        ldr_list.append(img)

    input_ldrs = np.concatenate(ldr_list, axis=-1)
    total_count += 1
    h5_file_name = h5_path + get_name(total_count)+'.h5'
    f = h5.File(h5_file_name, 'w')
    f['ldr'] = input_ldrs
    f['hdr'] = hdr
    f['expos'] = np.array(expos)
    return total_count

def prepare_training_dataset(training_scene_path, h5_path):
    scenes = os.listdir(training_scene_path)
    scenes.sort()
    count = 0
    for i, scene in enumerate(scenes):
        if len(scene) == 3:
            print(i, 'loading scene ' + scene, count)
            count = load_train_scene(training_scene_path+scene+'/', h5_path, count)
        else:
            continue

def prepare_test_dataset(test_scene_path, h5_path):
    count = 0
    paper_scene = os.listdir(test_scene_path+'PAPER/')
    paper_scene.sort()
    for i, scene in enumerate(paper_scene):
        print('loading paper scene ' + scene)
        count = load_test_scene(test_scene_path+'PAPER/'+scene+'/', h5_path, count)

    extra_scene = os.listdir(test_scene_path+'EXTRA/')
    extra_scene.sort()
    for i, scene in enumerate(extra_scene):
        print('loading extra scene ' + scene)
        count = load_test_scene(test_scene_path+'EXTRA/'+scene+'/', h5_path, count)
    

if __name__ == "__main__":
    prepare_training_dataset(TRAINING_SCENE_PATH, 'h5_data_raw/Training/')
    prepare_test_dataset(TEST_SCENE_PATH, 'h5_data_raw/Test/')


'''
img = cv2.imread('PAPER/BarbequeDay/HDRImg.hdr', flags = cv2.IMREAD_ANYDEPTH)
print(img.shape, img.dtype)
print(np.max(img))

print(read_expo('PAPER/BarbequeDay/exposure.txt'))
'''
