import config
import sys

sys.path.insert(0, config.mxnet_path)
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import provider
import os

class DummyIter(mx.io.DataIter):
    def __init__(self, batch_size, num_point, files, is_training=True):
        super(DummyIter, self).__init__(batch_size)

        # self.batch = mx.io.DataBatch(data=[mx.nd.zeros(self.data_shape)],
        #                              label=[mx.nd.zeros(self.label_shape)])

        self.batch_size = batch_size
        self.num_point = num_point
        self.num_batches = 0
        self.files = files
        self.is_train = is_training
        self.data_shape = (batch_size, num_point, 3)
        self.label_shape = (batch_size,)
        self.provide_data = [('data', self.data_shape)]
        self.provide_label = [('softmax_label', self.label_shape)]

        self.train_file_idxs = np.arange(0, len(files))
        np.random.shuffle(self.train_file_idxs)

        self.cur_file_idx = 0
        self.cur_batch = 0

        self.current_data = []
        self.current_label = []

        self.load_datafile(self.cur_file_idx)

    def next(self):
        if self.cur_batch < self.num_batches:
            start_idx = self.cur_batch * self.batch_size
            end_idx = (self.cur_batch + 1) * self.batch_size
            self.cur_batch += 1
            return self.get_batch(start_idx, end_idx)
        else:
            if self.cur_file_idx == len(self.train_file_idxs)-1:
                self.cur_batch = 0
                raise StopIteration
            else:
                self.cur_file_idx += 1
                self.cur_batch = 0
                self.load_datafile(self.cur_file_idx)
                start_idx = self.cur_batch * self.batch_size
                end_idx = (self.cur_batch + 1) * self.batch_size
                self.cur_batch += 1
                return self.get_batch(start_idx, end_idx)

    def get_batch(self, start_idx, end_idx):
        if self.is_train:
            # Augment batched point clouds by rotation and jittering
            rotated_data = provider.rotate_point_cloud(self.current_data[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)
            label = self.current_label[start_idx:end_idx]
            return mx.io.DataBatch(data=[nd.array(jittered_data)], label=[nd.array(label)],
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            data = self.current_data[start_idx:end_idx, :, :]
            label = self.current_label[start_idx:end_idx]
            return mx.io.DataBatch(data=[nd.array(data)], label=[nd.array(label)],
                                   provide_data=self.provide_data, provide_label=self.provide_label)


    def load_datafile(self, file_idx):
        self.current_data, self.current_label = provider.loadDataFile(self.files[self.train_file_idxs[file_idx]])
        self.current_data = self.current_data[:, 0:self.num_point, :]
        self.current_data, self.current_label, _ = provider.shuffle_data(self.current_data, np.squeeze(self.current_label))
        self.current_label = np.squeeze(self.current_label)
        file_size = self.current_data.shape[0]
        self.num_batches = file_size // self.batch_size

    def reset(self):
        np.random.shuffle(self.train_file_idxs)
        self.cur_file_idx = 0
        self.cur_batch = 0
        self.current_data = []
        self.current_label = []
        self.load_datafile(self.cur_file_idx)

def dummy_iterator(batch_size, num_point, train_files, val_files):
    train_iter = DummyIter(batch_size, num_point, train_files)
    val_iter = DummyIter(batch_size, num_point, val_files, is_training=False)
    num_examples = 9840
    return train_iter, val_iter, num_examples

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ModelNet40 official train/test split
    TRAIN_FILES = provider.getDataFiles( \
        os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
    TEST_FILES = provider.getDataFiles( \
        os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

    NUM_POINT = 1024

    train_iter, val_iter, _ = dummy_iterator(1, NUM_POINT, train_files=TRAIN_FILES, val_files=TEST_FILES)
    for iter in [train_iter, val_iter]:
        num_batch = 0
        for i in iter:
            # print(i)
            num_batch+=1
        print(num_batch)