import numpy as np
from utils import fit_warp_mtx

class Params:
    def __init__(self):
        self.tfrecordfile = '/home/zehao/PycharmProjects/learn-tensorflow/mnist.tfrecords'
        self.warpScale = {"pert": 0.25, "trans": 0.25}
        self.warpType = 'homography'  # translation, similarity, affine
        self.batch_size = 200
        self.max_steps = 10000
        self.summary_step = 100
        self.checkpoint_step = 1000
        self.gpu = 0
        self.train_dir = '/tmp/zehao/logs/STN/train0'  # log path

        # --- below are automatically set ---
        self.H, self.W = 28, 28
        self.canon4pts = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=np.float32)
        self.Im4pts = np.array([[0, 0], [0, self.H - 1], [self.W - 1, self.H - 1], [self.W - 1, 0]], dtype=np.float32)
        self.warpGTmtrx = fit_warp_mtx(self.canon4pts, self.Im4pts, "affine")

