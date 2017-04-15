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
from params import Params

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'MNIST',
                           """Currently only support MNIST dataset.""")
tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/zehao/logs/STN/train0',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Maximum number of batches to run.""")
tf.app.flags.DEFINE_integer('summary_step', 10,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 1000,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")


def train():
    """Train STN"""

    params = Params()
    with tf.Graph().as_default():
        model = STN(FLAGS.gpu, params)

        saver = tf.train.Saver(tf.global_variables())
        tfConfig = tf.ConfigProto(allow_soft_placement=True)
        tfConfig.gpu_options.allow_growth = True
        init = tf.global_variables_initializer()
        sess = tf.Session(config=tfConfig)
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
            if step % FLAGS.summary_step == 0:
                op_list = [
                    model.train_op, model.loss, summary_op
                ]
                _, loss_value, summary_str = sess.run(op_list)
                summary_writer.add_summary(summary_str, step)
                print('loss: {}'.format(loss_value))
            else:
                _, loss_value = sess.run([model.train_op, model.loss])

            if step % 10 == 0:
                format_str = ('%s: step %d, loss = %.2f')
                print(format_str % (datetime.now(), step, loss_value))
                sys.stdout.flush()

            # Save the model checkpoint periodically.
            if step % FLAGS.checkpoint_step == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
        #pass
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
