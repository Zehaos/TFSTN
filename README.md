# TFSTN
This is a pure tensorflow implementation of Spatial Transformer Networks.

<div align="center">
  <img src="http://i.imgur.com/gfqLV3f.png"><br><br>
</div>

## Features:
- TF API 1.0.0
- Good visualization.
- Easy to be intergrated.
- Tfrecord input pipeline

## Graph
<div align="center">
<img src="https://github.com/Zehaos/TFSTN/blob/master/graph.png"><br><br>
</div>

## Result
<div align="center">
<img src="https://github.com/Zehaos/TFSTN/blob/master/img_summary.png"><br><br>
</div>

## Usage

### Make mnist tfrecord
[script](https://github.com/Zehaos/learn-tensorflow/blob/master/make_tfrecord.py)

### Train
```
python train.py
```

### Visualize
```
tensorboard --logdir=/tmp/zehao/logs/STN/train
```

## TODO
- [x] Train
- [x] Visualization
- [x] Transform input images
- [x] Different learning between stn&cnn
- [x] Tfrecord pipeline
- [ ] Intergrate into face recognition task

## Reference
[Spatial Transformer Networks](https://arxiv.org/pdf/1506.02025.pdf)

[IC-STN](https://github.com/ericlin79119/IC-STN)
