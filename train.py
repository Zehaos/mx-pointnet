import logging, os
import sys

import config

sys.path.insert(0, config.mxnet_path)
import mxnet as mx
from core.solver import Solver
from core.metric import AccMetric, MatLossMetric

from symbol import *
from dataset import dummy_iterator
import provider


def main(config):
    # log file
    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='{}/{}.log'.format(log_dir, config.model_prefix),
                        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    # model folder
    model_dir = "./model"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # set up environment
    devs = [mx.gpu(int(i)) for i in config.gpu_list]
    kv = mx.kvstore.create(config.kv_store)

    # set up iterator and symbol
    # iterator
    # data list
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # ModelNet40 official train/test split
    TRAIN_FILES = provider.getDataFiles( \
        os.path.join(BASE_DIR, config.train_files))
    TEST_FILES = provider.getDataFiles( \
        os.path.join(BASE_DIR, config.val_files))
    train, val, num_examples = dummy_iterator(config.batch_size, config.num_points, TRAIN_FILES, TEST_FILES)

    data_names = ('data',)
    label_names = ('softmax_label',)
    data_shapes = [('data', (config.batch_size, config.num_points, 3))]
    label_shapes = [('softmax_label', (config.batch_size,))]

    if config.network == 'pointnet':
        symbol = eval(config.network)(num_classes=config.num_classes, batch_size=config.batch_size/len(config.gpu_list),
                                      num_points=config.num_points)
    # train
    lr_scheduler = mx.lr_scheduler.FactorScheduler(step=int(200000/config.batch_size), factor=config.lr_factor, stop_factor_lr=1e-05)

    optimizer_params = {'learning_rate': config.lr,
                        'lr_scheduler': lr_scheduler}
    optimizer = "adam"

    eval_metrics = mx.metric.CompositeEvalMetric()
    if config.dataset == "modelnet40":
        for m in [AccMetric, MatLossMetric]:
            eval_metrics.add(m())

    solver = Solver(symbol=symbol,
                    data_names=data_names,
                    label_names=label_names,
                    data_shapes=data_shapes,
                    label_shapes=label_shapes,
                    logger=logging,
                    context=devs)
    epoch_end_callback = mx.callback.do_checkpoint("./model/" + config.model_prefix)
    batch_end_callback = mx.callback.Speedometer(config.batch_size, config.frequent)
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)
    arg_params = None
    aux_params = None
    if config.retrain:
        _, arg_params, aux_params = mx.model.load_checkpoint("model/{}".format(config.model_load_prefix),
                                                             config.model_load_epoch)
    solver.fit(train_data=train,
               eval_data=val,
               eval_metric=eval_metrics,
               epoch_end_callback=epoch_end_callback,
               batch_end_callback=batch_end_callback,
               initializer=initializer,
               arg_params=arg_params,
               aux_params=aux_params,
               optimizer=optimizer,
               optimizer_params=optimizer_params,
               begin_epoch=config.begin_epoch,
               num_epoch=config.num_epoch,
               kvstore=kv)


if __name__ == '__main__':
    main(config)
