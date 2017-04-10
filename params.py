import numpy as np
from utils import fit_warp_mtx

class Params:
    def __init__(self):
        self.batchSize = 50
        #self.baseLR, self.baseLRST = args.lr, args.lrST
        # --- below are automatically set ---
        self.H, self.W = 28, 28
        self.canon4pts = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=np.float32)
        self.Im4pts = np.array([[0, 0], [0, self.H - 1], [self.W - 1, self.H - 1], [self.W - 1, 0]], dtype=np.float32)
        self.warpGTmtrx = fit_warp_mtx(self.canon4pts, self.Im4pts)
