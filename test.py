import os
import sys
import config

sys.path.insert(0, config.mxnet_path)
import mxnet as mx
import provider
from dataset import dummy_iterator

def main(config):
    symbol, arg_params, aux_params = mx.model.load_checkpoint('./model/' + config.model_load_prefix, config.model_load_epoch)

    model = mx.model.FeedForward(symbol, mx.gpu(0), arg_params=arg_params, aux_params=aux_params)
    kv = mx.kvstore.create(config.kv_store)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # ModelNet40 official train/test split
    TRAIN_FILES = provider.getDataFiles( \
        os.path.join(BASE_DIR, config.train_files))
    TEST_FILES = provider.getDataFiles( \
        os.path.join(BASE_DIR, config.val_files))
    _, val, _ = dummy_iterator(config.batch_size, config.num_points, TRAIN_FILES, TEST_FILES)
    print model.score(val)


if __name__ == '__main__':
    main(config)
