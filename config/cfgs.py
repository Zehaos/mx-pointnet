mxnet_path = '/home/zehao/PycharmProjects/mx-maskrcnn/incubator-mxnet/python'
gpu_list = [0]
dataset = "modelnet40"
model_prefix = "pointnet"
network = "pointnet"
model_load_prefix = model_prefix
model_load_epoch = 250
retrain = False

batch_size = 32
batch_size *= len(gpu_list)
kv_store = 'device'

# optimizer
lr = 0.001/32.0*batch_size
wd = 0.0001
momentum = 0.9
if dataset == "modelnet40":
    lr_factor = 0.7
begin_epoch = model_load_epoch if retrain else 0
num_epoch = 250
frequent = 50

# network config
if dataset == "modelnet40":
    num_classes = 40
    num_points = 1024
    train_files = 'data/modelnet40_ply_hdf5_2048/train_files.txt'
    val_files = 'data/modelnet40_ply_hdf5_2048/test_files.txt'
