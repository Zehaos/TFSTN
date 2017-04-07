import tensorflow as tf
import numpy as np
import scipy.linalg

def vec2mtx(vec):
    batch_size = tf.shape(vec)[0]
    I = tf.ones([batch_size])
    p1, p2, p3, p4, p5, p6, p7, p8 = tf.unstack(vec, axis=1)
    warp_mtx = tf.transpose(tf.stack([[I + p1, p2, p3],
                                      [p4, I + p5, p6],
                                      [p7, p8, I]]), perm=[2, 0, 1])
    return warp_mtx


def fit_warp_mtx(src4pts, des4pts):
    ptsN = len(src4pts)
    x, y, u, v, o, i = src4pts[:, 0], src4pts[:, 1], \
                       des4pts[:, 0], des4pts[:, 1], \
                       np.zeros([ptsN]), np.ones([ptsN])

    a = np.concatenate((np.stack([x, y, i, o, o, o], axis=1),
                        np.stack([o, o, o, x, y, i], axis=1)), axis=0)
    b = np.concatenate((u, v), axis=0)
    p = scipy.linalg.lstsq(a, b)[0].squeeze()
    warp_mtx = np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [0, 0, 1]], dtype=np.float32)
    return warp_mtx