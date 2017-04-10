import tensorflow as tf
import numpy as np
import scipy.linalg

def vec2mtx(pBatch, params):
    batchSize = tf.shape(pBatch)[0]
    O = tf.zeros([batchSize])
    I = tf.ones([batchSize])
    if params.warpType == "translation":
        tx, ty = tf.unstack(pBatch, axis=1)
        pMtrxBatch = tf.transpose(tf.stack([[I, O, tx],
                                            [O, I, ty],
                                            [O, O, I]]), perm=[2, 0, 1])
    elif params.warpType == "similarity":
        pc, ps, tx, ty = tf.unstack(pBatch, axis=1)
        pMtrxBatch = tf.transpose(tf.stack([[I + pc, -ps, tx],
                                            [ps, I + pc, ty],
                                            [O, O, I]]), perm=[2, 0, 1])
    elif params.warpType == "affine":
        p1, p2, p3, p4, p5, p6 = tf.unstack(pBatch, axis=1)
        pMtrxBatch = tf.transpose(tf.stack([[I + p1, p2, p3],
                                            [p4, I + p5, p6],
                                            [O, O, I]]), perm=[2, 0, 1])
    elif params.warpType == "homography":
        p1, p2, p3, p4, p5, p6, p7, p8 = tf.unstack(pBatch, axis=1)
        pMtrxBatch = tf.transpose(tf.stack([[I + p1, p2, p3],
                                            [p4, I + p5, p6],
                                            [p7, p8, I]]), perm=[2, 0, 1])
    return pMtrxBatch


def fit_warp_mtx(src4pts, des4pts, warpType):
    ptsN = len(src4pts)
    x, y, u, v, o, i = src4pts[:, 0], src4pts[:, 1], \
                       des4pts[:, 0], des4pts[:, 1], \
                       np.zeros([ptsN]), np.ones([ptsN])
    if warpType == "similarity":
        a = np.concatenate((np.stack([x, -y, i, o], axis=1),
                            np.stack([y, x, o, i], axis=1)), axis=0)
        b = np.concatenate((u, v), axis=0)
        p = scipy.linalg.lstsq(a, b)[0].squeeze()
        warp_mtx = np.array([[p[0], -p[1], p[2]], [p[1], p[0], p[3]], [0, 0, 1]], dtype=np.float32)
    elif warpType == "affine":
        a = np.concatenate((np.stack([x, y, i, o, o, o], axis=1),
                            np.stack([o, o, o, x, y, i], axis=1)), axis=0)
        b = np.concatenate((u, v), axis=0)
        p = scipy.linalg.lstsq(a, b)[0].squeeze()
        warp_mtx = np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [0, 0, 1]], dtype=np.float32)
    return warp_mtx