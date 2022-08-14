from tensorflow.keras import layers, models, backend
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
from common import *


def pretrain_vgg16():
    vgg_model = VGG16(include_top=False, input_shape=(None, None, 3))
    loss_model = models.Model(vgg_model.input, vgg_model.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return loss_model


def conv_model(nFeat, in_channels):
    x = layers.Input(shape=(None, None, in_channels))
    y = conv_lrelu(x, nFeat, 1)
    return models.Model(x, y)


def conv_model_multiscale(nFeat, in_channels):
    x = layers.Input(shape=(None, None, in_channels))
    x1 = Space2Depth(scale=2)(x)
    x2 = Space2Depth(scale=2)(x1)
    x3 = Space2Depth(scale=2)(x2)

    t = conv_lrelu(x, nFeat, 3)
    t1 = conv_lrelu(x1, nFeat, 3)
    t2 = conv_lrelu(x2, nFeat, 3)

    t3 = conv_lrelu(x3, nFeat, 3)
    return models.Model(x, [t, t1, t2, t3])


def conv_model_multiscale_5(nFeat, in_channels):
    x = layers.Input(shape=(None, None, in_channels))
    x1 = Space2Depth(scale=2)(x)
    x2 = Space2Depth(scale=2)(x1)
    x3 = Space2Depth(scale=2)(x2)
    x4 = Space2Depth(scale=2)(x3)
    print(x4.shape)
    t = conv_lrelu(x, nFeat, 3)
    t1 = conv_lrelu(x1, nFeat, 3)
    t2 = conv_lrelu(x2, nFeat, 3)
    t3 = conv_lrelu(x3, nFeat, 3)
    t4 = conv_lrelu(x4, nFeat, 3)
    return models.Model(x, [t, t1, t2, t3, t4])


def get_u(x):
    return tf.reduce_mean(x, axis=-1, keep_dims=True)


def get_c(x):
    return tf.reduce_sum(tf.square(x), axis=-1, keep_dims=True)


def get_s(x):
    _x, u = x
    return _x - u


def dot(x):
    x1, x2 = x
    return tf.reduce_sum(x1 * x2, axis=-1, keep_dims=True)


def merge(x):
    x1, x2, x3, w = x
    return x1 * w[:, :, :, 0::3] + x2 * w[:, :, :, 1::3] + x3 * w[:, :, :, 2::3]


def div(x):
    a, b = x
    return tf.div(a, b + 1e-12)


def Decomp(shape):
    x = layers.Input(shape=shape)
    u = layers.Lambda(lambda x: get_u(x))(x)
    s = layers.Lambda(lambda x: get_s(x))([x, u])
    c = layers.Lambda(lambda x: get_c(x))(s)
    return models.Model(x, [u, c, s])


def Recons(n_channels):
    def _recons(x):
        uc, s = x
        return uc[:, :, :, 1::2] * s + uc[:, :, :, 0::2]

    uc = layers.Input(shape=(None, None, 2))
    s = layers.Input(shape=(None, None, n_channels))
    y = layers.Lambda(lambda x: _recons(x))([uc, s])
    return models.Model([uc, s], y)


def normalize(x):
    u = tf.reduce_mean(x, axis=-1, keep_dims=True)
    s = tf.nn.l2_normalize(x - u, axis=-1)
    return s


def weighted_block(x, nFilters):
    t = conv_lrelu(x, nFilters, 1)
    t = conv_lrelu(t, nFilters, 3)
    t = conv(t, 3, 3)
    y = layers.Activation('softmax')(t)
    return y


def res_block(x, nFilters):
    t = conv_lrelu(x, nFilters, 3)
    t = conv(t, nFilters, 3)
    return layers.Add()([x, t])


def dense_block(x, nFilters):
    dilation_rates = [2, 2, 2, 2, 2, 2]
    t = x
    for d in dilation_rates:
        _t = conv_lrelu(t, nFilters // 2, 3, dilation_rate=d)
        t = layers.Concatenate(axis=-1)([t, _t])
    t = conv(t, nFilters, 1)
    return layers.Add()([x, t])


def s_block(ref_s, s, nFilters, nBlocks, nChannels, block_type='res'):
    if block_type == 'res':
        block = res_block
    if block_type == 'dense':
        block = dense_block

    _s = layers.Concatenate(axis=-1)([ref_s, s])
    _s = conv_lrelu(_s, nFilters, 1)
    for i in range(nBlocks):
        _s = block(_s, nFilters)
    _s = conv(_s, nChannels, 1)
    y = layers.Add()([ref_s, _s])
    return y


def uc_block(x, nFilters):
    t0 = conv_lrelu(x, nFilters, 3)
    t = conv_lrelu(t0, nFilters * 2, 3, strides=(2, 2))
    t = layers.GlobalAveragePooling2D()(t)
    t = layers.Dense(nFilters * 8, activation='relu')(t)
    t = layers.Dense(nFilters * 4, activation='relu')(t)
    t = layers.Dense(nFilters)(t)
    t = layers.Multiply()([t0, t])
    t = conv_lrelu(t, nFilters, 3)

    for i in range(5):
        _t = conv_lrelu(t, nFilters // 2, 3)
        t = layers.Concatenate(axis=-1)([t, _t])
    t = conv_lrelu(t, nFilters, 1)
    y = conv(t, 2, 3)
    return y




def modelv2(nFeat, in_channels, nBlocks, Res=False, use_vgg=False, vgg_model=None):
    #attention fusion before sigmoid
    def DRDB(x, growth, d_list, alpha=1):
        def _bp1(_x):
            t = conv(_x, 64, 3)
            t = adaptive_implicit_trans2(alpha=alpha)(t)
            t = conv(t, nFeat, 3)
            return t

        def _bp2(_x):
            t = conv(_x, nFeat, 3)
            # t = adadptive_implicit_trans2(alpha=alpha)(t)
            # t = conv(t, nFeat, 3)
            return t

        _bp = _bp1
        _x = conv_lrelu(x, nFeat, 3)
        for d in d_list:
            t = conv_lrelu(_x, growth, 3, dilation_rate=d)
            _x = layers.Concatenate(axis=-1)([_x, t])
        t1 = _bp(_x)
        t2 = _bp(_x)
        t3 = _bp(_x)
        t4 = _bp(_x)
        t = layers.Concatenate(axis=-1)([t1, t2, t3, t4])

        return layers.Add()([x, t])

    def attention_fusion_block(att, _att, alpha=0.5):
        def fuse(x, alpha):
            x1, x2 = x
            return x1 * alpha + x2 * (1 - alpha)
            
        _att_up = layers.UpSampling2D()(_att)
        att_fuse = layers.Lambda(lambda x: fuse(x, alpha))([_att_up, att])
        return att_fuse

    def multiscale_attention_block(f_x1, f1_x1, f2_x1, f3_x1,
                                   f_x2, f1_x2, f2_x2, f3_x2):

        f3 = layers.Concatenate(axis=-1)([f3_x1, f3_x2])
        f2 = layers.Concatenate(axis=-1)([f2_x1, f2_x2])
        f1 = layers.Concatenate(axis=-1)([f1_x1, f1_x2])
        f = layers.Concatenate(axis=-1)([f_x1, f_x2])

        att_f3 = conv(f3, 1, 3)
        att_f3_x1x2 = layers.Activation('sigmoid')(att_f3)
        f3_x1x2 = layers.Multiply()([att_f3_x1x2, f3_x1])

        att_f2 = conv(f2, 1, 3)
        att_f2 = attention_fusion_block(att_f2, att_f3)
        att_f2_x1x2 = layers.Activation('sigmoid')(att_f2)
        f2_x1x2 = layers.Multiply()([att_f2_x1x2, f2_x1])


        att_f1 = conv(f1, 1, 3)
        att_f1 = attention_fusion_block(att_f1, att_f2)
        att_f1_x1x2 = layers.Activation('sigmoid')(att_f1)
        f1_x1x2 = layers.Multiply()([att_f1_x1x2, f1_x1])

        att_f = conv(f, 1, 3)
        att_f = attention_fusion_block(att_f, att_f1)
        att_f_x1x2 = layers.Activation('sigmoid')(att_f)
        f_x1x2 = layers.Multiply()([att_f_x1x2, f_x1])

        return f_x1x2, f1_x1x2, f2_x1x2, f3_x1x2


    def bandpass_branch(x, nBlocks, alpha):
        concat_list = []
        t = x
        for i in range(nBlocks):
            t = DRDB(t, nFeat // 2, (1, 2, 3, 2, 1), alpha)
        return t

    def baseline(f, nBlocks, alpha, up=False):
        f_wav = WaveletConvLayer()(f)
        _f_wav = bandpass_branch(f_wav, nBlocks, 1.0)
        _f = WaveletInvLayer()(_f_wav)
        _f = layers.Add()([f, _f])

        if up:
            _f = conv(_f, nFeat * 4, 3)
            _f = Depth2Space(scale=2)(_f)
        return _f

    x1 = layers.Input(shape=(None, None, in_channels))
    x2 = layers.Input(shape=(None, None, in_channels))
    x3 = layers.Input(shape=(None, None, in_channels))
    output_list = []

    cmm = conv_model_multiscale(nFeat, in_channels)
    f_x1, f1_x1, f2_x1, f3_x1 = cmm(x1)
    f_x2, f1_x2, f2_x2, f3_x2 = cmm(x2)
    f_x3, f1_x3, f2_x3, f3_x3 = cmm(x3)

    f_x1x2, f1_x1x2, f2_x1x2, f3_x1x2 = multiscale_attention_block(
        f_x1, f1_x1, f2_x1, f3_x1,
        f_x2, f1_x2, f2_x2, f3_x2)
    f_x3x2, f1_x3x2, f2_x3x2, f3_x3x2 = multiscale_attention_block(
        f_x3, f1_x3, f2_x3, f3_x3,
        f_x2, f1_x2, f2_x2, f3_x2)

    f3 = layers.Concatenate(axis=-1)([f3_x1x2, f3_x2, f3_x3x2])
    f3 = conv_lrelu(f3, nFeat, 1)
    f2 = layers.Concatenate(axis=-1)([f2_x1x2, f2_x2, f2_x3x2])
    f2 = conv_lrelu(f2, nFeat, 1)
    f1 = layers.Concatenate(axis=-1)([f1_x1x2, f1_x2, f1_x3x2])
    f1 = conv_lrelu(f1, nFeat, 1)
    f = layers.Concatenate(axis=-1)([f_x1x2, f_x2, f_x3x2])
    f = conv_lrelu(f, nFeat, 1)

    # f3 = DRDB(f3, nFeat//2, 6, f_ref=f3_x2, Res=Res)
    f3 = baseline(f3, nBlocks, 1.0)
    f3 = deconv_lrelu(f3, nFeat)
    f3 = deconv_lrelu(f3, nFeat)
    f3 = deconv_lrelu(f3, nFeat)
    f3 = conv(f3, 3, 3)

    # f2 = DRDB(f2, nFeat//2, 6, f_ref=f2_x2, Res=Res)
    f2 = baseline(f2, nBlocks, 1.0)
    f2 = deconv_lrelu(f2, nFeat)
    f2 = deconv_lrelu(f2, nFeat)
    f2 = conv(f2, 3, 3)

    # f1 = DRDB(f1, nFeat//2, 6, f_ref=f1_x2, Res=Res)
    f1 = baseline(f1, nBlocks, 1.0)
    f1 = deconv_lrelu(f1, nFeat)
    f1 = conv(f1, 3, 3)

    # f = DRDB(f, nFeat//2, 6, f_ref=f_x2, Res=Res)
    f = baseline(f, nBlocks, 1.0)
    f = conv(f, 3, 3)
    f = layers.Add()([f3, f2, f1, f])
    y = layers.Activation('sigmoid')(f)

    if use_vgg:
        return models.Model([x1, x2, x3], [y, vgg_model(y)])
    else:
        return models.Model([x1, x2, x3], y)
