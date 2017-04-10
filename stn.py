from nn_skeleton import ModelSkeleton
import tensorflow as tf
import numpy as np
from utils import fit_warp_mtx, vec2mtx


class STN(ModelSkeleton):
    def __init__(self, gpu_id, params):
        self.params = params
        with tf.device('/gpu:{}'.format(gpu_id)):
            ModelSkeleton.__init__(self, params.batchSize, params.W, params.H)
            self._add_forward_graph()
            self._add_loss_graph()
            self._add_train_graph()

    def _add_forward_graph(self):
        """NN architecture."""
        stn1 = self._stn_module('stn1', self.image_input)

        conv2 = self._conv_layer('conv2', stn1, filters=30, size=1, stride=1,
                                 padding='VALID', channels=1)
        fc3 = self._fc_layer('fc3', conv2, 10, flatten=True, xavier=True)
        self.preds = fc3

    def _stn_module(self, layer_name, images):
        tf.summary.image('image_input', images, max_outputs=self.params.batchSize)

        conv1 = self._conv_layer(
            layer_name + '/conv1', images, filters=4, size=1, stride=1,
            padding='VALID')
        conv2 = self._conv_layer(
            layer_name + '/conv2', conv1, filters=8, size=1, stride=1,
            padding='VALID')
        pool2 = self._pooling_layer(
            layer_name + '/pool2', conv2, size=2, stride=2, padding='VALID'
        )
        fc3 = self._fc_layer(
            layer_name + '/fc3', pool2, 100, flatten=True)
        fc4 = self._fc_layer(
            layer_name + '/fc4', fc3, 8, flatten=False, relu=False)
        #fc4 = tf.Print(fc4, [tf.split(fc4, self.params.batchSize, axis=0)[0][0]])
        with tf.variable_scope(layer_name+'/ImWarp') as scope:
            warp_mtx = vec2mtx(fc4)
            images_warped = self._warp_op(images, warp_mtx)
            tf.summary.image(layer_name + 'image_warped', images_warped, max_outputs=self.params.batchSize)
        return images_warped

    def _warp_op(self, images, warp_mtx):
        H, W = 28, 28
        canon4pts = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=np.float32)
        img4pts = np.array([[0, 0], [0, H - 1], [W - 1, H - 1], [W - 1, 0]], dtype=np.float32)
        warp_gt_mtx = fit_warp_mtx(canon4pts, img4pts)

        #batch_size = tf.shape(images)[0]
        warpGTmtrxBatch = tf.tile(tf.expand_dims(warp_gt_mtx, 0), [self.params.batchSize, 1, 1])
        transMtrxBatch = tf.matmul(warpGTmtrxBatch, warp_mtx)
        # warp the canonical coordinates
        X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
        XYhom = tf.transpose(tf.stack([X.reshape([-1]), Y.reshape([-1]), np.ones([X.size])], axis=1))
        XYhomBatch = tf.tile(tf.expand_dims(XYhom, 0), [self.params.batchSize, 1, 1])
        XYwarpHomBatch = tf.matmul(transMtrxBatch, tf.to_float(XYhomBatch))
        XwarpHom, YwarpHom, ZwarpHom = tf.split(XYwarpHomBatch, 3, 1)
        Xwarp = tf.reshape(XwarpHom / ZwarpHom, [self.params.batchSize, self.params.H, self.params.W])
        Ywarp = tf.reshape(YwarpHom / ZwarpHom, [self.params.batchSize, self.params.H, self.params.W])
        # get the integer sampling coordinates
        Xfloor, Xceil = tf.floor(Xwarp), tf.ceil(Xwarp)
        Yfloor, Yceil = tf.floor(Ywarp), tf.ceil(Ywarp)
        XfloorInt, XceilInt = tf.to_int32(Xfloor), tf.to_int32(Xceil)
        YfloorInt, YceilInt = tf.to_int32(Yfloor), tf.to_int32(Yceil)
        ImIdx = tf.tile(tf.reshape(tf.range(self.params.batchSize), [self.params.batchSize, 1, 1]),
                        [1, self.params.H, self.params.W])
        ImVecBatch = tf.reshape(images, [-1, tf.shape(images)[3]])
        ImVecBatchOutside = tf.concat([ImVecBatch, tf.zeros([1, tf.shape(images)[3]])], 0)
        idxUL = (ImIdx * H + YfloorInt) * W + XfloorInt
        idxUR = (ImIdx * H + YfloorInt) * W + XceilInt
        idxBL = (ImIdx * H + YceilInt) * W + XfloorInt
        idxBR = (ImIdx * H + YceilInt) * W + XceilInt
        idxOutside = tf.fill([self.params.batchSize, self.params.H, self.params.W],
                             self.params.batchSize * self.params.H * self.params.W)

        def insideIm(Xint, Yint):
            return (Xint >= 0) & (Xint < W) & (Yint >= 0) & (Yint < H)

        idxUL = tf.where(insideIm(XfloorInt, YfloorInt), idxUL, idxOutside)
        idxUR = tf.where(insideIm(XceilInt, YfloorInt), idxUR, idxOutside)
        idxBL = tf.where(insideIm(XfloorInt, YceilInt), idxBL, idxOutside)
        idxBR = tf.where(insideIm(XceilInt, YceilInt), idxBR, idxOutside)
        # bilinear interpolation
        Xratio = tf.reshape(Xwarp - Xfloor, [self.params.batchSize, self.params.H, self.params.W, 1])
        Yratio = tf.reshape(Ywarp - Yfloor, [self.params.batchSize, self.params.H, self.params.W, 1])
        ImUL = tf.to_float(tf.gather(ImVecBatchOutside, idxUL)) * (1 - Xratio) * (1 - Yratio)
        ImUR = tf.to_float(tf.gather(ImVecBatchOutside, idxUR)) * (Xratio) * (1 - Yratio)
        ImBL = tf.to_float(tf.gather(ImVecBatchOutside, idxBL)) * (1 - Xratio) * (Yratio)
        ImBR = tf.to_float(tf.gather(ImVecBatchOutside, idxBR)) * (Xratio) * (Yratio)
        ImWarpBatch = ImUL + ImUR + ImBL + ImBR
        ImWarpBatch = tf.identity(ImWarpBatch)

        return ImWarpBatch



