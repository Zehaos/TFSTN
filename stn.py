from nn_skeleton import ModelSkeleton
import tensorflow as tf
import numpy as np
from utils import vec2mtx


class STN(ModelSkeleton):
    def __init__(self, gpu_id, params):
        with tf.device('/gpu:{}'.format(gpu_id)):
            ModelSkeleton.__init__(self, params)
            self._add_datagen_graph()
            self._add_forward_graph()
            self._add_loss_graph()
            self._add_train_graph()

    def _add_datagen_graph(self):
        with tf.variable_scope('Gen_imgs') as scope:
            # init tfrecord reader
            self.reader = tf.TFRecordReader()
            self.filename_queue = tf.train.string_input_producer([self.params.tfrecordfile],
                                                                 shuffle=True)
            _, serialized = self.reader.read(self.filename_queue)
            features = tf.parse_single_example(
                serialized,
                features={
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'pixels': tf.FixedLenFeature([], tf.int64),
                    'label': tf.FixedLenFeature([], tf.int64)
                }
            )

            image = tf.reshape(tf.decode_raw(features['image_raw'], tf.uint8), [28, 28, 1])
            image = tf.cast(image, tf.float32)
            label = tf.one_hot(features['label'], 10)
            self.image_input, self.labels = tf.train.shuffle_batch(
                [image, label], batch_size=self.params.batch_size, capacity=10000 + 3 * self.params.batch_size,
                min_after_dequeue=10000
            )


    def _add_forward_graph(self):
        """NN architecture."""
        stn1 = self._stn_module('Stn1', self.image_input)

        conv2 = self._conv_layer('Conv2', stn1, filters=3, size=9, stride=1,
                                 padding='VALID', channels=1, stddev=0.03)
        fc3 = self._fc_layer('Fc3', conv2, 10, flatten=True, xavier=True, relu=False,  stddev=0.03)
        self.preds = fc3

    def _stn_module(self, layer_name, images):
        with tf.variable_scope('Tsf_imgs') as scope:
            perturbations = self._gen_perturbations(self.params)
            warp_mtx = vec2mtx(perturbations, self.params)
            images = self._warp_op(images, warp_mtx)
            images = tf.reshape(images, [self.params.batch_size, self.params.H, self.params.W, 1])

        tf.summary.image('image_input', images, max_outputs=10)

        conv1 = self._conv_layer(
            layer_name + '/conv1', images, filters=4, size=7, stride=2,
            padding='VALID', stddev=0.01)
        conv2 = self._conv_layer(
            layer_name + '/conv2', conv1, filters=8, size=5, stride=2,
            padding='VALID', stddev=0.01)
        pool2 = self._pooling_layer(
            layer_name + '/pool2', conv2, size=2, stride=2, padding='VALID'
        )
        fc3 = self._fc_layer(
            layer_name + '/fc3', pool2, 48, flatten=True, stddev=0.01)
        fc4 = self._fc_layer(
            layer_name + '/fc4', fc3, 8, flatten=False, relu=False, stddev=0.01)

        with tf.variable_scope(layer_name+'/ImWarp') as scope:
            warp_mtx = vec2mtx(fc4, self.params)
            images_warped = self._warp_op(images, warp_mtx)
            tf.summary.image(layer_name + 'image_warped', images_warped, max_outputs=10)
        return images_warped

    # Thanks @ericlin79119 IC-STN
    def _warp_op(self, images, warp_mtx):
        H, W = self.params.H, self.params.W
        warp_gt_mtx = self.params.warpGTmtrx

        warpGTmtrxBatch = tf.tile(tf.expand_dims(warp_gt_mtx, 0), [self.params.batch_size, 1, 1])
        transMtrxBatch = tf.matmul(warpGTmtrxBatch, warp_mtx)
        # warp the canonical coordinates
        X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
        XYhom = tf.transpose(tf.stack([X.reshape([-1]), Y.reshape([-1]), np.ones([X.size])], axis=1))
        XYhomBatch = tf.tile(tf.expand_dims(XYhom, 0), [self.params.batch_size, 1, 1])
        XYwarpHomBatch = tf.matmul(transMtrxBatch, tf.to_float(XYhomBatch))
        XwarpHom, YwarpHom, ZwarpHom = tf.split(XYwarpHomBatch, 3, 1)
        Xwarp = tf.reshape(XwarpHom / ZwarpHom, [self.params.batch_size, self.params.H, self.params.W])
        Ywarp = tf.reshape(YwarpHom / ZwarpHom, [self.params.batch_size, self.params.H, self.params.W])
        # get the integer sampling coordinates
        Xfloor, Xceil = tf.floor(Xwarp), tf.ceil(Xwarp)
        Yfloor, Yceil = tf.floor(Ywarp), tf.ceil(Ywarp)
        XfloorInt, XceilInt = tf.to_int32(Xfloor), tf.to_int32(Xceil)
        YfloorInt, YceilInt = tf.to_int32(Yfloor), tf.to_int32(Yceil)
        ImIdx = tf.tile(tf.reshape(tf.range(self.params.batch_size), [self.params.batch_size, 1, 1]),
                        [1, self.params.H, self.params.W])
        ImVecBatch = tf.reshape(images, [-1, tf.shape(images)[3]])
        ImVecBatchOutside = tf.concat([ImVecBatch, tf.zeros([1, tf.shape(images)[3]])], 0)
        idxUL = (ImIdx * H + YfloorInt) * W + XfloorInt
        idxUR = (ImIdx * H + YfloorInt) * W + XceilInt
        idxBL = (ImIdx * H + YceilInt) * W + XfloorInt
        idxBR = (ImIdx * H + YceilInt) * W + XceilInt
        idxOutside = tf.fill([self.params.batch_size, self.params.H, self.params.W],
                             self.params.batch_size * self.params.H * self.params.W)

        def insideIm(Xint, Yint):
            return (Xint >= 0) & (Xint < W) & (Yint >= 0) & (Yint < H)

        idxUL = tf.where(insideIm(XfloorInt, YfloorInt), idxUL, idxOutside)
        idxUR = tf.where(insideIm(XceilInt, YfloorInt), idxUR, idxOutside)
        idxBL = tf.where(insideIm(XfloorInt, YceilInt), idxBL, idxOutside)
        idxBR = tf.where(insideIm(XceilInt, YceilInt), idxBR, idxOutside)
        # bilinear interpolation
        Xratio = tf.reshape(Xwarp - Xfloor, [self.params.batch_size, self.params.H, self.params.W, 1])
        Yratio = tf.reshape(Ywarp - Yfloor, [self.params.batch_size, self.params.H, self.params.W, 1])
        ImUL = tf.to_float(tf.gather(ImVecBatchOutside, idxUL)) * (1 - Xratio) * (1 - Yratio)
        ImUR = tf.to_float(tf.gather(ImVecBatchOutside, idxUR)) * (Xratio) * (1 - Yratio)
        ImBL = tf.to_float(tf.gather(ImVecBatchOutside, idxBL)) * (1 - Xratio) * (Yratio)
        ImBR = tf.to_float(tf.gather(ImVecBatchOutside, idxBR)) * (Xratio) * (Yratio)
        ImWarpBatch = ImUL + ImUR + ImBL + ImBR
        ImWarpBatch = tf.identity(ImWarpBatch)

        return ImWarpBatch

    # generate jitter
    # Thanks @ericlin79119 IC-STN
    def _gen_perturbations(self, params):
        X = np.tile(params.canon4pts[:, 0], [params.batch_size, 1])
        Y = np.tile(params.canon4pts[:, 1], [params.batch_size, 1])
        dX = tf.random_normal([params.batch_size, 4]) * params.warpScale["pert"] \
             + tf.random_normal([params.batch_size, 1]) * params.warpScale["trans"]
        dY = tf.random_normal([params.batch_size, 4]) * params.warpScale["pert"] \
             + tf.random_normal([params.batch_size, 1]) * params.warpScale["trans"]
        O = np.zeros([params.batch_size, 4], dtype=np.float32)
        I = np.ones([params.batch_size, 4], dtype=np.float32)
        # fit warp parameters to generated displacements
        if params.warpType == "affine":
            J = np.concatenate([np.stack([X, Y, I, O, O, O], axis=-1),
                                np.stack([O, O, O, X, Y, I], axis=-1)], axis=1)
            dXY = tf.expand_dims(tf.concat(1, [dX, dY]), -1)
            dpBatch = tf.matrix_solve_ls(J, dXY)  # J*Output=dXY
        elif params.warpType == "homography":
            A = tf.concat([tf.stack([X, Y, I, O, O, O, -X * (X + dX), -Y * (X + dX)], axis=-1),
                           tf.stack([O, O, O, X, Y, I, -X * (Y + dY), -Y * (Y + dY)], axis=-1)], 1)
            b = tf.expand_dims(tf.concat([X + dX, Y + dY], 1), -1)
            dpBatch = tf.matrix_solve_ls(A, b)
            dpBatch -= tf.to_float(tf.reshape([1, 0, 0, 0, 1, 0, 0, 0], [1, 8, 1]))
        dpBatch = tf.reduce_sum(dpBatch, reduction_indices=-1)
        return dpBatch



