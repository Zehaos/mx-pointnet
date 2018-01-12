import mxnet as mx
import numpy as np

from mx_constant import MyConstant

eps = 1e-5

def input_transform_net(data, batch_size, num_points, workspace, bn_mom=0.9, scope="itn_"):
    data = mx.sym.expand_dims(data, axis=1)  # (32,1,1024,3)
    conv0 = mx.sym.Convolution(data=data, num_filter=64, kernel=(1, 3), stride=(1, 1), name=scope + "conv0",
                               workspace=workspace)
    conv0 = mx.sym.Activation(data=conv0, act_type='relu', name=scope + 'relu0')
    conv0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, momentum=bn_mom, name=scope + 'bn0')

    conv1 = mx.sym.Convolution(data=conv0, num_filter=128, kernel=(1, 1), stride=(1, 1), name=scope + "conv1",
                               workspace=workspace)
    conv1 = mx.sym.Activation(data=conv1, act_type='relu', name=scope + 'relu1')
    conv1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, momentum=bn_mom, name=scope + 'bn1')

    conv2 = mx.sym.Convolution(data=conv1, num_filter=1024, kernel=(1, 1), stride=(1, 1), name=scope + "conv2",
                               workspace=workspace)
    conv2 = mx.sym.Activation(data=conv2, act_type='relu', name=scope + 'relu2')
    conv2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps, momentum=bn_mom, name=scope + 'bn2')

    pool3 = mx.sym.Pooling(data=conv2, kernel=(num_points, 1), pool_type='max', name=scope + 'pool3')
    pool3_reshaped = mx.sym.Reshape(data=pool3, shape=(batch_size, -1))

    fc4 = mx.sym.FullyConnected(data=pool3_reshaped, num_hidden=512, name=scope + 'fc4')
    fc4 = mx.sym.Activation(data=fc4, act_type='relu', name=scope + 'relu4')
    fc4 = mx.sym.BatchNorm(data=fc4, fix_gamma=False, eps=eps, momentum=bn_mom, name=scope + 'bn4')

    fc5 = mx.sym.FullyConnected(data=fc4, num_hidden=256, name=scope + 'fc5')
    fc5 = mx.sym.Activation(data=fc5, act_type='relu', name=scope + 'relu5')
    fc5 = mx.sym.BatchNorm(data=fc5, fix_gamma=False, eps=eps, momentum=bn_mom, name=scope + 'bn5')

    input_transformer_weight = mx.sym.Variable(name="input_transformer_weight", shape=(9, 256), init=mx.init.Zero())
    input_transformer_bias = mx.sym.Variable(name="input_transformer_bias", shape=(9), init=mx.init.Zero())
    transform = mx.sym.FullyConnected(data=fc5, num_hidden=9, weight=input_transformer_weight, bias=input_transformer_bias, name=scope + 'fc6')

    const_arr = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    a = mx.sym.Variable('itn_addi_bias', shape=(batch_size, 9), init=MyConstant(value=[const_arr]*batch_size))
    a = mx.sym.BlockGrad(a)  # now variable a is a constant

    transform = mx.sym.elemwise_add(transform, a, name=scope + "add_eye")

    transform_reshaped = mx.sym.Reshape(data=transform, shape=(batch_size, 3, 3), name=scope + "reshape_transform")
    return transform_reshaped


def feature_transform_net(data, batch_size, num_points, workspace, bn_mom=0.9, scope="ftn_"):
    conv0 = mx.sym.Convolution(data=data, num_filter=64, kernel=(1, 1), stride=(1, 1), name=scope + "conv0",
                               workspace=workspace)
    conv0 = mx.sym.Activation(data=conv0, act_type='relu', name=scope + 'relu0')
    conv0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, momentum=bn_mom, name=scope + 'bn0')

    conv1 = mx.sym.Convolution(data=conv0, num_filter=128, kernel=(1, 1), stride=(1, 1), name=scope + "conv1",
                               workspace=workspace)
    conv1 = mx.sym.Activation(data=conv1, act_type='relu', name=scope + 'relu1')
    conv1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, momentum=bn_mom, name=scope + 'bn1')

    conv2 = mx.sym.Convolution(data=conv1, num_filter=1024, kernel=(1, 1), stride=(1, 1), name=scope + "conv2",
                               workspace=workspace)
    conv2 = mx.sym.Activation(data=conv2, act_type='relu', name=scope + 'relu2')
    conv2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps, momentum=bn_mom, name=scope + 'bn2')

    pool3 = mx.sym.Pooling(data=conv2, kernel=(num_points, 1), pool_type='max', name=scope + 'pool3')
    pool3_reshaped = mx.sym.Reshape(data=pool3, shape=(batch_size, -1))

    fc4 = mx.sym.FullyConnected(data=pool3_reshaped, num_hidden=512, name=scope + 'fc4')
    fc4 = mx.sym.Activation(data=fc4, act_type='relu', name=scope + 'relu4')
    fc4 = mx.sym.BatchNorm(data=fc4, fix_gamma=False, eps=eps, momentum=bn_mom, name=scope + 'bn4')

    fc5 = mx.sym.FullyConnected(data=fc4, num_hidden=256, name=scope + 'fc5')
    fc5 = mx.sym.Activation(data=fc5, act_type='relu', name=scope + 'relu5')
    fc5 = mx.sym.BatchNorm(data=fc5, fix_gamma=False, eps=eps, momentum=bn_mom, name=scope + 'bn5')

    feat_transformer_weight = mx.sym.Variable(name="feat_transformer_weight", shape=(64*64, 256), init=mx.init.Zero())
    feat_transformer_bias = mx.sym.Variable(name="feat_transformer_bias", shape=(64*64), init=mx.init.Zero())

    transform = mx.sym.FullyConnected(data=fc5, num_hidden=64 * 64, weight=feat_transformer_weight, bias=feat_transformer_bias, name=scope + 'fc6')

    const_arr = np.eye(64, dtype=np.float32).flatten().tolist()
    a = mx.sym.Variable('ftn_addi_bias', shape=(batch_size, 64 * 64), init=MyConstant(value=[const_arr]*batch_size))
    a = mx.sym.BlockGrad(a)  # now variable a is a constant

    transform = mx.sym.elemwise_add(transform, a, name=scope + "add_eye")
    transform_reshaped = mx.sym.Reshape(data=transform, shape=(batch_size, 64, 64), name=scope + "reshape_transform")
    return transform_reshaped
