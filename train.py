from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import sys
import traceback
import time

import numpy as np
import tensorflow as tf
from stn import STN

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'MNIST',
                           """Currently only support MNIST dataset.""")
tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/zehao/logs/STN/train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Maximum number of batches to run.""")
tf.app.flags.DEFINE_integer('summary_step', 10,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 1000,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")


# load MNIST data
def loadMNIST(fname):
    if not os.path.exists(fname):
        # download and preprocess MNIST dataset
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        trainData, validData, testData = {}, {}, {}
        trainData["image"] = mnist.train.images.reshape([-1, 28, 28]).astype(np.float32)
        validData["image"] = mnist.validation.images.reshape([-1, 28, 28]).astype(np.float32)
        testData["image"] = mnist.test.images.reshape([-1, 28, 28]).astype(np.float32)
        trainData["label"] = mnist.train.labels.astype(np.float32)
        validData["label"] = mnist.validation.labels.astype(np.float32)
        testData["label"] = mnist.test.labels.astype(np.float32)
        os.makedirs(os.path.dirname(fname))
        np.savez(fname, train=trainData, valid=validData, test=testData)
    MNIST = np.load(fname)
    trainData = MNIST["train"].item()
    validData = MNIST["valid"].item()
    testData = MNIST["test"].item()
    return trainData, validData, testData


# generate training batch
def genPerturbations(params):
    with tf.variable_scope("genPerturbations"):
        X = np.tile(params.canon4pts[:, 0], [params.batchSize, 1])
        Y = np.tile(params.canon4pts[:, 1], [params.batchSize, 1])
        dX = tf.random_normal([params.batchSize, 4]) * params.warpScale["pert"] \
             + tf.random_normal([params.batchSize, 1]) * params.warpScale["trans"]
        dY = tf.random_normal([params.batchSize, 4]) * params.warpScale["pert"] \
             + tf.random_normal([params.batchSize, 1]) * params.warpScale["trans"]
        O = np.zeros([params.batchSize, 4], dtype=np.float32)
        I = np.ones([params.batchSize, 4], dtype=np.float32)
        # fit warp parameters to generated displacements
        A = tf.concat([tf.stack([X, Y, I, O, O, O, -X * (X + dX), -Y * (X + dX)], axis=-1),
                       tf.stack([O, O, O, X, Y, I, -X * (Y + dY), -Y * (Y + dY)], axis=-1)], 1)
        b = tf.expand_dims(tf.concat([X + dX, Y + dY], 1), -1)
        dpBatch = tf.matrix_solve_ls(A, b)
        dpBatch -= tf.to_float(tf.reshape([1, 0, 0, 0, 1, 0, 0, 0], [1, 8, 1]))
        dpBatch = tf.reduce_sum(dpBatch, reduction_indices=-1)
    return dpBatch


def train():
    """Train STN"""
    # load data
    print("loading MNIST dataset...")
    trainData, validData, testData = loadMNIST("data/MNIST.npz")
    batch_size = 50
    with tf.Graph().as_default():
        model = STN(FLAGS.gpu)

        saver = tf.train.Saver(tf.global_variables())

        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(init)

        initial_step = 0
        global_step = model.global_step

        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            initial_step = global_step.eval(session=sess)

        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        summary_op = tf.summary.merge_all()

        for step in xrange(initial_step, FLAGS.max_steps):
            start_time = time.time()

            # generate training data
            rand_idx = np.random.randint(len(trainData["image"]), size=batch_size)
            image_per_batch = trainData["image"][rand_idx]
            label_per_batch = trainData["label"][rand_idx]
            image_per_batch = np.reshape(image_per_batch, [batch_size, 28, 28, 1])
            feed_dict = {
                model.image_input: image_per_batch,
                model.labels: label_per_batch,
            }

            if step % FLAGS.summary_step == 0:
                op_list = [
                    model.train_op, model.loss, summary_op
                ]
                _, loss_value, summary_str = sess.run(op_list, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                print('loss: {}'.format(loss_value))
            else:
                _, loss_value = sess.run([model.train_op, model.loss],
                                         feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % 10 == 0:
                num_images_per_step = batch_size
                images_per_sec = num_images_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f images/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    images_per_sec, sec_per_batch))
                sys.stdout.flush()

            # Save the model checkpoint periodically.
            if step % FLAGS.checkpoint_step == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        # tf.gfile.DeleteRecursively(FLAGS.train_dir)
        pass
    else:
        tf.gfile.MakeDirs(FLAGS.train_dir)
    try:
        train()
    except:
        print
        "Exception in user code:"
        print
        '-' * 60
        traceback.print_exc(file=sys.stdout)
        print
        '-' * 60


if __name__ == '__main__':
    tf.app.run()
