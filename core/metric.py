import mxnet as mx
import numpy as np
from mxnet.metric import check_label_shapes

class AccMetric(mx.metric.EvalMetric):
    def __init__(self, axis=0):
        super(AccMetric, self).__init__('Acc')
        self.axis = axis

    def update(self, labels, preds):
        labels = labels[0]
        preds = preds[0]
        check_label_shapes(labels, preds)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = mx.nd.argmax(pred_label, axis=self.axis)

            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')

            check_label_shapes(label, pred_label)

            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)


class MatLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(MatLossMetric, self).__init__('MatLoss')

    def update(self, labels, preds):

        labels = labels[0]
        loss = preds[1]

        pred_label = labels.asnumpy()
        loss = loss.asnumpy()
        self.sum_metric += np.sum(loss)
        self.num_inst += len(pred_label.flat)