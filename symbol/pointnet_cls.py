import sys
import config

sys.path.insert(0, config.mxnet_path)
import mxnet as mx
import numpy as np

from mx_constant import MyConstant
from transform_nets import input_transform_net, feature_transform_net

eps = 1e-5

def pointnet(num_classes, batch_size, num_points, bn_mom=0.9, workspace=512, scope="pointnet_"):
    point_cloud = mx.sym.Variable(name='data') # (B,P,3)

    # Point cloud transformer
    transform = input_transform_net(point_cloud, batch_size, num_points, workspace, bn_mom, scope=scope + "itn_") # (B, 3, 3)
    point_cloud_transformed = mx.sym.batch_dot(point_cloud, transform, name=scope + "input_transform")
    input_image = mx.sym.expand_dims(point_cloud_transformed, axis=1) # (B, 1, P, 3)

    # Shared mlp
    conv0 = mx.sym.Convolution(data=input_image, num_filter=64, kernel=(1, 3), stride=(1, 1), name=scope + "conv0",
                               workspace=workspace)
    conv0 = mx.sym.Activation(data=conv0, act_type='relu', name=scope + 'relu0')
    conv0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, momentum=bn_mom, name=scope + 'bn0')

    conv1 = mx.sym.Convolution(data=conv0, num_filter=64, kernel=(1, 1), stride=(1, 1), name=scope + "conv1",
                               workspace=workspace)
    conv1 = mx.sym.Activation(data=conv1, act_type='relu', name=scope + 'relu1')
    conv1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, momentum=bn_mom, name=scope + 'bn1') # (B, 64, 1024, 1)

    # Feature transformer
    transform = feature_transform_net(conv1, batch_size, num_points, workspace, bn_mom, scope=scope + "ftn_") # (B, 64, 64)
    conv1_reshaped = mx.sym.Reshape(conv1, (-1, 64, num_points), name=scope + "conv1_reshape")  # (B, 64, 1024)
    conv1_reshaped = mx.sym.transpose(conv1_reshaped, axes=(0,2,1), name=scope + "conv1_reshape_transpose")
    conv1_transformed = mx.sym.batch_dot(conv1_reshaped, transform, name=scope + "conv1_transform")
    conv1_transformed = mx.sym.swapaxes(conv1_transformed, 1, 2, name=scope + "conv1_swapaxes")
    conv1_transformed = mx.sym.expand_dims(conv1_transformed, axis=3, name=scope + "conv1_expanddim")

    conv2 = mx.sym.Convolution(data=conv1_transformed, num_filter=64, kernel=(1, 1), stride=(1, 1),
                               name=scope + "conv2",
                               workspace=workspace)
    conv2 = mx.sym.Activation(data=conv2, act_type='relu', name=scope + 'relu2')
    conv2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps, momentum=bn_mom, name=scope + 'bn2')

    conv3 = mx.sym.Convolution(data=conv2, num_filter=128, kernel=(1, 1), stride=(1, 1), name=scope + "conv3",
                               workspace=workspace)
    conv3 = mx.sym.Activation(data=conv3, act_type='relu', name=scope + 'relu3')
    conv3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=eps, momentum=bn_mom, name=scope + 'bn3')

    conv4 = mx.sym.Convolution(data=conv3, num_filter=1024, kernel=(1, 1), stride=(1, 1), name=scope + "conv4",
                               workspace=workspace)
    conv4 = mx.sym.Activation(data=conv4, act_type='relu', name=scope + 'relu4')
    conv4 = mx.sym.BatchNorm(data=conv4, fix_gamma=False, eps=eps, momentum=bn_mom, name=scope + 'bn4')

    pool5 = mx.sym.Pooling(data=conv4, kernel=(num_points, 1), pool_type='max', name=scope + 'pool5')
    pool5_reshaped = mx.sym.Reshape(data=pool5, shape=(batch_size, -1), name=scope + 'pool5_reshape')

    fc6 = mx.sym.FullyConnected(data=pool5_reshaped, num_hidden=512, name=scope + 'fc6')
    fc6 = mx.sym.Activation(data=fc6, act_type='relu', name=scope + 'relu6')
    fc6 = mx.sym.BatchNorm(data=fc6, fix_gamma=False, eps=eps, momentum=bn_mom, name=scope + 'bn6')
    fc6 = mx.sym.Dropout(fc6, p=0.7)

    fc7 = mx.sym.FullyConnected(data=fc6, num_hidden=256, name=scope + 'fc7')
    fc7 = mx.sym.Activation(data=fc7, act_type='relu', name=scope + 'relu7')
    fc7 = mx.sym.BatchNorm(data=fc7, fix_gamma=False, eps=eps, momentum=bn_mom, name=scope + 'bn7')
    fc7 = mx.sym.Dropout(fc7, p=0.7)

    fc8 = mx.sym.FullyConnected(data=fc7, num_hidden=40, name=scope + 'fc8')
    cls = mx.sym.SoftmaxOutput(data=fc8, name='softmax')
    
    transform_transposed = mx.sym.transpose(transform, axes=(0,2,1), name=scope+"transpose_transform")
    mat_diff = mx.sym.batch_dot(transform, transform_transposed, name=scope+"transform_dot")
    const_arr = np.eye(64, dtype=np.float32).tolist()
    a = mx.sym.Variable('addition_loss_constant', shape=(batch_size, 64, 64), init=MyConstant(value=[const_arr]*batch_size))
    a = mx.sym.BlockGrad(a)  # now variable a is a constant
    mat_diff = mx.sym.elemwise_sub(mat_diff, a, name=scope + "sub_eye")
    mat_diff_loss = mx.sym.sum(mx.sym.square(mat_diff))
    matloss = mx.sym.make_loss(name='transform_mat_loss', data=mat_diff_loss, grad_scale=0.001 / (batch_size*2.0))

    return mx.sym.Group([cls, matloss])
