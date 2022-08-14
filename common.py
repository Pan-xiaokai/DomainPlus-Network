from tensorflow.python.keras import layers, backend, losses, initializers, models, constraints
from tensorflow.python.keras import backend as k
from tensorflow.python.keras.utils import conv_utils
import tensorflow as tf
from tensorflow.python.ops import array_ops
from math import log, sqrt, cos, sin, pi
from deform_conv import *
from ciede2000 import *

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.util.tf_export import tf_export
import numpy as np
import cv2

GAMMA = 2.2

def conv_lrelu(x, filters, kernel, padding='same', use_bias = False, dilation_rate=1, strides=(1,1)):
    if dilation_rate == 0:
        y = layers.Conv2D(filters,1,padding=padding,use_bias=use_bias)(x)
        y = layers.LeakyReLU()(y)
    else:
        y = layers.Conv2D(filters,kernel,padding=padding,use_bias=use_bias,
            dilation_rate=dilation_rate,
            strides=strides)(x)
        y = layers.LeakyReLU()(y)
    return y

def conv(x, filters, kernel, padding='same', use_bias=True, dilation_rate=1, strides = (1,1)):
    y = layers.Conv2D(filters,kernel,padding=padding,use_bias=use_bias,
        dilation_rate=dilation_rate, strides=strides)(x)
    return y

def conv_relu(x, filters, kernel, padding='same', use_bias = False, dilation_rate=1, strides=(1,1)):
    if dilation_rate == 0:
        y = layers.Conv2D(filters,1,padding=padding,use_bias=use_bias, activation='relu')(x)
    else:
        y = layers.Conv2D(filters,kernel,padding=padding,use_bias=use_bias,
            dilation_rate=dilation_rate,
            strides=strides,
            activation='relu')(x)
    return y

def deconv_relu(x, filter):
    y = layers.Conv2DTranspose(filter,4,strides=2,padding='same',activation='relu')(x)
    return y

def deconv_lrelu(x, filter):
    y = layers.Conv2DTranspose(filter,4,strides=2,padding='same')(x)
    y = layers.LeakyReLU()(y)
    return y

def deform_conv_block(x, filter, act=None):
    offset = conv(x, 18, 3)
    _x = layers.Concatenate(axis=-1)([offset, x])
    y = DeformableConv2d(filter)(_x)
    if act != None:
        y = layers.Activation(act)(y)
    return y


def LDR2HDR(LDR, expo):
    return (LDR**GAMMA)/expo

def u_law(x):
    t1 = k.log(1+5000*x)
    t2 = 1/log(1+5000)
    return t1*t2

def u_law_numpy(x):
    t1 = np.log(1+5000*x)
    t2 = 1/log(1+5000)
    return t1*t2

def new_law(x):
    t1 = x - (k.log(1+5000*x) / 5000)
    t2 = 1/log(1+5000)
    return t1*t2

def new_law_numpy(x):
    t1 = x - (np.log(1+5000*x) / 5000)
    t2 = 1/log(1+5000)
    return t1*t2

def u_loss_attention(y_true, y_pred):
    #print(y_pred.shape)
    _img = y_pred[:,:,:,0:3]
    att = tf.reduce_mean(y_pred[:,:,:,3:5], axis=-1, keepdims=True)
    _pred = u_law(_img)
    _true = u_law(y_true)
    return tf.reduce_mean(tf.abs(_pred-_true), axis=-1)+tf.reduce_mean(tf.multiply(tf.abs(_pred-_true),att), axis=-1)

def u_ssim_loss(y_pred, y_true):
    pred_ = u_law(y_pred)
    pred_y = tf.image.rgb_to_yuv(pred_)[:,:,:,0::3]
    true_ = u_law(y_true)
    true_y = tf.image.rgb_to_yuv(true_)[:,:,:,0::3]
    ssim = tf.image.ssim(pred_y,true_y,1.0)
    mae_loss = tf.reduce_mean(tf.abs(pred_-true_),axis=(-1,-2,-3))
    return 1-ssim+mae_loss

def u_loss(y_pred, y_true):
    pred_ = u_law(y_pred)
    true_ = u_law(y_true)
    return losses.mae(true_, pred_)

def new_loss(y_pred, y_true):
    pred_ = new_law(y_pred)
    true_ = new_law(y_true)
    return losses.mae(true_, pred_)

def MAE(y_pred,y_true):
    return losses.mae(y_true,y_pred)

def u_loss_clip(y_true, y_pred):
    _pred = k.clip(y_pred, 0, 1)
    pred_ = u_law(_pred)
    true_ = u_law(y_true)
    return losses.mae(true_, pred_)

def linear_fitting_3D_points(points):

    Sum_X = 0.0
    Sum_Y = 0.0
    Sum_Z = 0.0
    Sum_XZ = 0.0
    Sum_YZ = 0.0
    Sum_Z2 = 0.0

    for i in range(0, len(points)):
        xi = points[i][0]
        yi = points[i][1]
        zi = points[i][2]

        Sum_X = Sum_X + xi
        Sum_Y = Sum_Y + yi
        Sum_Z = Sum_Z + zi
        Sum_XZ = Sum_XZ + xi * zi
        Sum_YZ = Sum_YZ + yi * zi
        Sum_Z2 = Sum_Z2 + zi ** 2

    n = len(points)
    den = n * Sum_Z2 - Sum_Z * Sum_Z
    k1 = (n * Sum_XZ - Sum_X * Sum_Z) / den
    b1 = (Sum_X - k1 * Sum_Z) / n
    k2 = (n * Sum_YZ - Sum_Y * Sum_Z) / den
    b2 = (Sum_Y - k2 * Sum_Z) / n

    return k1, b1, k2, b2

def get_distance_from_point_to_line(line_point1, line_point2, point):
    x1 = line_point1[0]
    y1 = line_point1[1]
    z1 = line_point1[2]
    x2 = line_point2[0]
    y2 = line_point2[1]
    z2 = line_point2[2]
    x3 = point[0]
    y3 = point[1]
    z3 = point[2]

    t1 = (x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1) + (z2 - z1) * (z3 - z1)
    t2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 -z1) * (z2 - z1)
    t = t1 / t2

    x4 = (x2 - x1) * t + x1
    y4 = (y2 - y1) * t + y1
    z4 = (z2 - z1) * t + z1

    d = np.sqrt((x4 - x3) * (x4 - x3) + (y4 - y3) * (y4 - y3) + (z4 - z3) * (z4 - z3))

    return d

def calc_d(y_pred, y_true, size):
    input_shape = k.int_shape(y_pred)
    batch, width, length, depth = input_shape
    cut_width = size
    cut_length = size

    h_count = int(width / cut_width)
    w_count = int(length / cut_length)
    count = 0
    d_all = 0

    for i in range(0, h_count):
        for j in range(0, w_count):
            count += 1
            gt_1 = y_true[:, i * cut_width: (i + 1) * cut_width, j * cut_length: (j + 1) * cut_length, :]
            pr_1 = y_pred[:, i * cut_width: (i + 1) * cut_width, j * cut_length: (j + 1) * cut_length, :]

            gt_2 = tf.reshape(gt_1, (gt_1.shape[0], cut_width * cut_length,3))
            pr_2 = tf.reshape(pr_1, (pr_1.shape[0], cut_width * cut_length,3))

            # gt_2 = gt_1.reshape((cut_width * cut_length, 3))
            # pr_2 = pr_1.reshape((cut_width * cut_length, 3))
            model = KMeans(n_clusters=2)
            model.fit(gt_2)
            gt_centers = model.cluster_centers_
            # print(gt_centers)
            model.fit(pr_2)
            pr_centers = model.cluster_centers_

            d = get_distance_from_point_to_line(gt_centers[0], gt_centers[1], pr_centers[0])
            d_all += d

    distance = d_all / count

    return distance

def hdr_merge(im1, im2, im3, expos):
    a1 = np.zeros(im2.shape,dtype=np.float32)
    a2 = np.zeros(im2.shape,dtype=np.float32)
    a3 = np.zeros(im2.shape,dtype=np.float32)
    a1[im2>=0.5] = 2-2*im2[im2>=0.5]
    a1[im2<0.5]  = 1
    a1 = 1-a1
    a2[im2>=0.5] = 2-2*im2[im2>=0.5]
    a2[im2<0.5]  = 2*im2[im2<0.5]
    a3[im2>=0.5] = 1
    a3[im2<0.5] = 2*im2[im2<0.5]
    a3 = 1-a3
    #print(np.max(a1), np.max(a2), np.max(a3))
    #return (im1*a1/expos[0]+im2*a2/expos[1]+im3*a3/expos[2])/(a1+a2+a3)
    return (a1*LDR2HDR(im1,expos[0])+a2*LDR2HDR(im2,expos[1])+a3*LDR2HDR(im3,expos[2]))/(a1+a2+a3)

class Sign(layers.Layer):
    def __init__(self, **kwargs):
        super(Sign, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return tf.sign(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


class Space2Depth(layers.Layer):
	def __init__(self, scale, **kwargs):
		super(Space2Depth, self).__init__(**kwargs)
		self.scale = scale

	def call(self, inputs, **kwargs):
		return array_ops.space_to_depth(inputs, self.scale)

	def compute_output_shape(self, input_shape):
		if input_shape[1] != None and input_shape[2] != None:
			return (None, int(input_shape[1]/self.scale), int(input_shape[2]/self.scale), input_shape[3]*self.scale**2)
		else:
			return (None, None, None, input_shape[3]*self.scale**2)

class Depth2Space(layers.Layer):
	def __init__(self, scale, **kwargs):
		super(Depth2Space, self).__init__(**kwargs)
		self.scale = scale
	def call(self, inputs, **kwargs):
		return array_ops.depth_to_space(inputs, self.scale)

	def compute_output_shape(self, input_shape):
		if input_shape[1] != None and input_shape[2] != None:
			return (None, input_shape[1]*self.scale, input_shape[2]*self.scale, int(input_shape[3]/self.scale**2))
		else:
			return (None, None, None, int(input_shape[3]/self.scale**2))

class Sample(layers.Layer):
    def __init__(self, window, strides = 1, padding='valid', **kwargs):
        super(Sample, self).__init__(**kwargs)
        self.window = window
        self.strides = strides
        self.padding = padding
        _pad_size = (window-1)//2
        self.padding_size = pad_sizes = [[0, 0], [_pad_size, _pad_size], [_pad_size, _pad_size], [0, 0]]

    def build(self, input_shape):
        window = self.window
        kernel = np.zeros((window, window, 1, window*window),dtype=np.float32)
        for i in range(window*window):
            kernel[int(i/window),int(i%window),0,i]=1
        self.kernel = k.variable(value=kernel)
        self.kernel = array_ops.tile(self.kernel, [1, 1, input_shape[-1], 1])

    def call(self, inputs, **kwargs):
        x = array_ops.pad(inputs,self.padding_size, mode='REFLECT')
        y = k.depthwise_conv2d(x, self.kernel, padding=self.padding, strides=(self.strides,self.strides))
        return y

    def compute_output_shape(self, input_shape):
        h = conv_utils.conv_output_length(input_shape[1],self.window, 'valid', self.strides)
        w = conv_utils.conv_output_length(input_shape[2],self.window, 'valid', self.strides)
        return (None,)+ tuple([h, w]) + (self.window*self.window*input_shape[3],)

class SampleBack(layers.Layer):
    def __init__(self, window, padding='valid', strides=1, output_channels=1, **kwargs):
        super(SampleBack, self).__init__(**kwargs)
        self.window = window
        self.padding = padding
        self.output_channels = output_channels
        self.strides = strides

    def build(self, input_shape):
        window = self.window
        kernel = np.zeros((window, window, 1, window*window),dtype=np.float32)
        for i in range(window*window):
            kernel[int(i/window),int(i%window),0,i]=1
        self.kernel = k.variable(value=kernel)

    def call(self, inputs, **kwargs):
        window = self.window
        input_shape = k.shape(inputs)
        int_shape = k.int_shape(inputs)
        print(int_shape)
        #print(input_shape[-1],window*window*self.output_channels)
        assert int_shape[-1] == window*window*self.output_channels
        h = conv_utils.deconv_length(input_shape[1],self.strides,self.window,self.padding,None)
        w = conv_utils.deconv_length(input_shape[2],self.strides,self.window,self.padding,None)

        if self.output_channels == 1:
            y = k.conv2d_transpose(inputs, self.kernel, strides=(self.strides,self.strides), padding= self.padding, output_shape=(input_shape[0],h,w,1))
        else:
            output_list = []
            step = window**2
            for j in range(self.output_channels):
                output_list.append(k.conv2d_transpose(inputs[:,:,:,step*j:step*(j+1)], self.kernel, strides=(self.strides,self.strides), padding= self.padding, output_shape=(input_shape[0],h,w,1)))
            y = k.concatenate(output_list, axis=-1)
        return y

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        c_axis, h_axis, w_axis = 3, 1, 2
        kernel_h = self.window
        kernel_w = self.window
        stride_h = self.strides
        stride_w = self.strides

        output_shape[c_axis] = self.output_channels
        output_shape[h_axis] = conv_utils.deconv_length(
            output_shape[h_axis], stride_h, kernel_h, self.padding,None)
        output_shape[w_axis] = conv_utils.deconv_length(
            output_shape[w_axis], stride_w, kernel_w, self.padding,None)
        return tuple(output_shape)

class ToneMapping(layers.Layer):
    def __init__(self, **kwargs):
        super(ToneMapping, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(1,1,1,input_shape[-1]),
                        initializer = initializers.get('ones'),
                        name='mapper')

    def call(self, inputs):
        kernel = tf.exp(self.kernel)
        return inputs * kernel

    def compute_output_shape(self, input_shape):
        return input_shape

class BicubicLayer(layers.Layer):
    def __init__(self, size, **kwargs):
        self.size = size
        super(BicubicLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.image.resize(inputs,self.size, method=tf.image.ResizeMethod.BICUBIC)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],)+self.size+(input_shape[-1],)

class ScaleLayer(layers.Layer):
    def __init__(self, s, **kwargs):
        self.s = s
        super(ScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape = (1,),
            name = 'scale',
            initializer=initializers.Constant(value=self.s),
            constraint=constraints.NonNeg())
    def call(self, inputs):
        return inputs*self.kernel

class ChannelSplit(layers.Layer):
    def __init__(self, nFeat, index, **kwargs):
        self.nFeat= nFeat
        self.index = index
        super(ChannelSplit, self).__init__(**kwargs)

    def call(self, inputs):
        start = self.nFeat * self.index
        end = start + self.nFeat
        return inputs[:,:,:,start:end]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.nFeat)

class adaptive_implicit_trans2(layers.Layer):
    def __init__(self, size = 8, alpha = 1, **kwargs):
        self.size = size
        self.alpha = alpha
        super(adaptive_implicit_trans2, self).__init__(**kwargs)

    def build(self, input_shape):
        size = int(self.size)
        total = size**2
        conv_shape = (1,1,total,total)
        self.it_weights = self.add_weight(
            shape = (1,1,total,1),
            initializer = initializers.Constant(value=self.alpha),
            #constraint = constraints.NonNeg(),
            name = 'ait_conv')
        kernel = np.zeros(conv_shape)
        r1 = sqrt(1.0/size)
        r2 = sqrt(2.0/size)
        for i in range(size):
            _u = 2*i+1
            for j in range(size):
                _v = 2*j+1
                index = i*size+j
                for u in range(size):
                    for v in range(size):
                        index2 = u*size+v
                        t = cos(_u*u*pi/size/2)*cos(_v*v*pi/size/2)
                        t = t*r1 if u==0 else t*r2
                        t = t*r1 if v==0 else t*r2
                        kernel[0,0,index2,index] = t
        self.kernel = k.variable(value = kernel, dtype = 'float32')

    def call(self, inputs):
        kernel = self.kernel*self.it_weights
        y = k.conv2d(inputs,
                        kernel,
                        padding = 'same',
                        data_format='channels_last')
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

class tile_layer(layers.Layer):
    def __init__(self, scale=2, **kwargs):
        self.scale = scale
        super(tile_layer, self).__init__(**kwargs)

    def call(self, inputs):
        n = self.scale**2
        return array_ops.tile(inputs, [1,1,1,n])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3]*self.scale**2)

class WaveletConvLayer(layers.Layer):
    def __init__(self, **kwargs):
        self.scale=2
        super(WaveletConvLayer, self).__init__(**kwargs)

    def call(self, inputs):
        im_c1 = inputs[:, 0::2, 0::2, :] # 1
        im_c2 = inputs[:, 0::2, 1::2, :] # right up
        im_c3 = inputs[:, 1::2, 0::2, :] # left down
        im_c4 = inputs[:, 1::2, 1::2, :] # right right

        LL = im_c1 + im_c2 + im_c3 + im_c4
        LH = -im_c1 - im_c2 + im_c3 + im_c4
        HL = -im_c1 + im_c2 - im_c3 + im_c4
        HH = im_c1 - im_c2 - im_c3 + im_c4

        result = tf.concat([LL, LH, HL, HH], 3)
        return result

    def compute_output_shape(self, input_shape):
        if input_shape[1] != None and input_shape[2] != None:
            return (None, int(input_shape[1]/self.scale), int(input_shape[2]/self.scale), input_shape[3]*self.scale**2)
        else:
            return (None, None, None, input_shape[3]*self.scale**2)

class WaveletInvLayer(layers.Layer):
    def __init__(self, **kwargs):
        self.scale=2
        super(WaveletInvLayer, self).__init__(**kwargs)


    def call(self, inputs):
        sz = k.int_shape(inputs)
        inputs = inputs/4
        a = sz[-1]//4
        LL = inputs[:, :, :, 0:a]
        LH = inputs[:, :, :, a:2*a]
        HL = inputs[:, :, :, 2*a:3*a]
        HH = inputs[:, :, :, 3*a:]

        aa = LL - LH - HL + HH
        bb = LL - LH + HL - HH
        cc = LL + LH - HL - HH
        dd = LL + LH + HL + HH
        concated = tf.concat([aa, bb, cc, dd], axis=-1)
        reconstructed = array_ops.depth_to_space(concated, 2)
        return reconstructed

    def compute_output_shape(self, input_shape):
        if input_shape[1] != None and input_shape[2] != None:
            return (None, input_shape[1]*self.scale, input_shape[2]*self.scale, int(input_shape[3]/self.scale**2))
        else:
            return (None, None, None, int(input_shape[3]/self.scale**2))


def bicubic_model():
    x = layers.Input(shape=(256,256,3))
    _x2 = BicubicLayer((128,128))(x)
    _x4 = BicubicLayer((64,64))(x)
    return models.Model(x,[_x4,_x2])

def sobel_edges(image):
    """Returns a tensor holding Sobel edge maps.

    Arguments:
    image: Image tensor with shape [batch_size, h, w, d] and type float32 or
    float64.  The image(s) must be 2x2 or larger.

    Returns:
    Tensor holding edge maps for each channel. Returns a tensor with shape
    [batch_size, h, w, d, 2] where the last two dimensions hold [[dy[0], dx[0]],
    [dy[1], dx[1]], ..., [dy[d-1], dx[d-1]]] calculated using the Sobel filter.
    """

    # Define vertical and horizontal Sobel filters.
    static_image_shape = image.get_shape()
    image_shape = array_ops.shape(image)
    kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
                [[0, -1, -2], [1, 0, -1], [2, 1, 0]]]
    num_kernels = len(kernels)
    kernels = np.transpose(np.asarray(kernels), (1, 2, 0))
    kernels = np.expand_dims(kernels, -2)
    kernels_tf = constant_op.constant(kernels, dtype=image.dtype)
    kernels_tf = array_ops.tile(kernels_tf, [1, 1, image_shape[-1], 1],
                                name='sobel_filters')

    # Use depth-wise convolution to calculate edge maps per channel.
    pad_sizes = [[0, 0], [1, 1], [1, 1], [0, 0]]
    padded = array_ops.pad(image, pad_sizes, mode='REFLECT')

    # Output tensor has shape [batch_size, h, w, d * num_kernels].
    strides = [1, 1, 1, 1]
    output = nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')
    # Reshape to [batch_size, h, w, d, num_kernels].
    shape = array_ops.concat([image_shape, [num_kernels]], 0)
    output = array_ops.reshape(output, shape=shape)
    output.set_shape(static_image_shape.concatenate([num_kernels]))
    return output

def sobel_edges_d2(image):
    """Returns a tensor holding Sobel edge maps.

    Arguments:
    image: Image tensor with shape [batch_size, h, w, d] and type float32 or
    float64.  The image(s) must be 2x2 or larger.

    Returns:
    Tensor holding edge maps for each channel. Returns a tensor with shape
    [batch_size, h, w, d, 2] where the last two dimensions hold [[dy[0], dx[0]],
    [dy[1], dx[1]], ..., [dy[d-1], dx[d-1]]] calculated using the Sobel filter.
    """

    # Define vertical and horizontal Sobel filters.
    static_image_shape = image.get_shape()
    image_shape = array_ops.shape(image)
    kernels = [[[-1, 0, -2, 0, -1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 2, 0, 1]],
                [[-1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [-2, 0, 0, 0, 2], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 1]],
                [[-2, 0, -1, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 2]],
                [[0, 0, -1, 0, -2], [0, 0, 0, 0, 0], [1, 0, 0, 0, -1], [0, 0, 0, 0, 0], [2, 0, 1, 0, 0]]]
    num_kernels = len(kernels)
    kernels = np.transpose(np.asarray(kernels), (1, 2, 0))
    kernels = np.expand_dims(kernels, -2)
    kernels_tf = constant_op.constant(kernels, dtype=image.dtype)
    kernels_tf = array_ops.tile(kernels_tf, [1, 1, image_shape[-1], 1],
                                name='sobel_filters')

    # Use depth-wise convolution to calculate edge maps per channel.
    pad_sizes = [[0, 0], [2, 2], [2, 2], [0, 0]]
    padded = array_ops.pad(image, pad_sizes, mode='REFLECT')

    # Output tensor has shape [batch_size, h, w, d * num_kernels].
    strides = [1, 1, 1, 1]
    output = nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')
    # Reshape to [batch_size, h, w, d, num_kernels].
    shape = array_ops.concat([image_shape, [num_kernels]], 0)
    output = array_ops.reshape(output, shape=shape)
    output.set_shape(static_image_shape.concatenate([num_kernels]))
    return output

def sobel_edges_d3(image):
    """Returns a tensor holding Sobel edge maps.

    Arguments:
    image: Image tensor with shape [batch_size, h, w, d] and type float32 or
    float64.  The image(s) must be 2x2 or larger.

    Returns:
    Tensor holding edge maps for each channel. Returns a tensor with shape
    [batch_size, h, w, d, 2] where the last two dimensions hold [[dy[0], dx[0]],
    [dy[1], dx[1]], ..., [dy[d-1], dx[d-1]]] calculated using the Sobel filter.
    """

    # Define vertical and horizontal Sobel filters.
    static_image_shape = image.get_shape()
    image_shape = array_ops.shape(image)
    kernels = [[[-1, 0, 0, -2, 0, 0, -1], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 2, 0, 0, 1]],
                [[-1, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [-2, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 1]],
                [[-2, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 2]],
                [[0, 0, 0, -1, 0, 0, -2], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 1, 0, 0, 0]]]
    num_kernels = len(kernels)
    kernels = np.transpose(np.asarray(kernels), (1, 2, 0))
    kernels = np.expand_dims(kernels, -2)
    kernels_tf = constant_op.constant(kernels, dtype=image.dtype)
    kernels_tf = array_ops.tile(kernels_tf, [1, 1, image_shape[-1], 1],
                                name='sobel_filters')

    # Use depth-wise convolution to calculate edge maps per channel.
    pad_sizes = [[0, 0], [3, 3], [3, 3], [0, 0]]
    padded = array_ops.pad(image, pad_sizes, mode='REFLECT')

    # Output tensor has shape [batch_size, h, w, d * num_kernels].
    strides = [1, 1, 1, 1]
    output = nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')
    # Reshape to [batch_size, h, w, d, num_kernels].
    shape = array_ops.concat([image_shape, [num_kernels]], 0)
    output = array_ops.reshape(output, shape=shape)
    output.set_shape(static_image_shape.concatenate([num_kernels]))
    return output

def sobel_edges_d4(image):
    """Returns a tensor holding Sobel edge maps.

    Arguments:
    image: Image tensor with shape [batch_size, h, w, d] and type float32 or
    float64.  The image(s) must be 2x2 or larger.

    Returns:
    Tensor holding edge maps for each channel. Returns a tensor with shape
    [batch_size, h, w, d, 2] where the last two dimensions hold [[dy[0], dx[0]],
    [dy[1], dx[1]], ..., [dy[d-1], dx[d-1]]] calculated using the Sobel filter.
    """

    # Define vertical and horizontal Sobel filters.
    static_image_shape = image.get_shape()
    image_shape = array_ops.shape(image)
    kernels = [[[-1, 0, 0, 0, -2, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 2, 0, 0, 0, 1]],
                [[-1, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [-2, 0, 0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 0, 0, 1]],
                [[-2, 0, 0, 0, -1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 2]],
                [[0, 0, 0, 0, -1, 0, 0, 0, -2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 1, 0, 0, 0, 0]]]
    num_kernels = len(kernels)
    kernels = np.transpose(np.asarray(kernels), (1, 2, 0))
    kernels = np.expand_dims(kernels, -2)
    kernels_tf = constant_op.constant(kernels, dtype=image.dtype)
    kernels_tf = array_ops.tile(kernels_tf, [1, 1, image_shape[-1], 1],
                                name='sobel_filters')

    # Use depth-wise convolution to calculate edge maps per channel.
    pad_sizes = [[0, 0], [4, 4], [4, 4], [0, 0]]
    padded = array_ops.pad(image, pad_sizes, mode='REFLECT')

    # Output tensor has shape [batch_size, h, w, d * num_kernels].
    strides = [1, 1, 1, 1]
    output = nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')
    # Reshape to [batch_size, h, w, d, num_kernels].
    shape = array_ops.concat([image_shape, [num_kernels]], 0)
    output = array_ops.reshape(output, shape=shape)
    output.set_shape(static_image_shape.concatenate([num_kernels]))
    return output

def loss_func_v3(y_true, y_pred):
    mae = losses.MAE
    def sobel_process(y_true, y_pred, sobel_func):
        sobel_pred = sobel_func(y_pred)*0.25
        sobel_true = sobel_func(y_true)*0.25
        dx_loss = mae(sobel_pred[:,:,:,:,0],sobel_true[:,:,:,:,0])
        dy_loss = mae(sobel_pred[:,:,:,:,1],sobel_true[:,:,:,:,1])
        dr_loss = mae(sobel_pred[:,:,:,:,2],sobel_true[:,:,:,:,2])
        dl_loss = mae(sobel_pred[:,:,:,:,3],sobel_true[:,:,:,:,3])
        return dx_loss+dy_loss+dr_loss+dl_loss
    y_true = u_law(y_true)
    y_pred = u_law(y_pred)
    sobel_d1_loss = sobel_process(y_true, y_pred, sobel_edges)
    sobel_d2_loss = sobel_process(y_true, y_pred, sobel_edges_d2)
    sobel_d3_loss = sobel_process(y_true, y_pred, sobel_edges_d3)
    sobel_d4_loss = sobel_process(y_true, y_pred, sobel_edges_d4)
    mae_loss = mae(y_true, y_pred)
    return mae_loss+sobel_d1_loss+sobel_d2_loss+sobel_d3_loss
