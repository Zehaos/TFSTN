import tensorflow as tf
import params

def _variable_on_device(name, shape, initializer, trainable=True):
    """Helper to create a Variable.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    dtype = tf.float32
    if not callable(initializer):
        var = tf.get_variable(name, initializer=initializer, trainable=trainable)
    else:
        var = tf.get_variable(
            name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    return var


def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    var = _variable_on_device(name, shape, initializer, trainable)
    if wd is not None and trainable:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


class ModelSkeleton:
    """Base class of NN detection models."""

    def __init__(self, params):
        self.image_input = tf.placeholder(
            tf.float32, [params.batchSize, params.H, params.W, 1],
            name='image_input'
        )
        self.labels = tf.placeholder(
            tf.float32, [params.batchSize, 10], name='labels'
        )
        # model parameters
        self.model_params = []
        self.params = params

    def _add_forward_graph(self):
        """NN architecture specification."""
        raise NotImplementedError

    def _add_interpretation_graph(self):
        """Interpret NN output."""
        raise NotImplementedError

    def _add_loss_graph(self):
        """Define the loss operation."""
        with tf.variable_scope('class_regression') as scope:
            softmax_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.preds,
                                                                  labels=self.labels)
            self.loss = tf.reduce_mean(softmax_loss)
            tf.summary.scalar('loss', self.loss)

    def _add_train_graph(self):
        """Define the training operation."""
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        lr = tf.train.exponential_decay(0.001,
                                        self.global_step,
                                        10000,
                                        0.5,
                                        staircase=True)

        tf.summary.scalar('learning_rate', lr)

        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        grads_vars = opt.compute_gradients(self.loss, tf.trainable_variables())

        with tf.variable_scope('clip_gradient') as scope:
            for i, (grad, var) in enumerate(grads_vars):
                grads_vars[i] = (tf.clip_by_norm(grad, 1.0), var)

        apply_gradient_op = opt.apply_gradients(grads_vars, global_step=self.global_step)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        for grad, var in grads_vars:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        with tf.control_dependencies([apply_gradient_op]):
            self.train_op = tf.no_op(name='train')

    def _add_viz_graph(self):
        """Define the visualization operation."""
        self.image_to_show = tf.placeholder(
            tf.float32, [None, self.params.H, self.params.W, 1],
            name='image_input'
        )
        self.viz_op = tf.summary.image('image_input',
                                       self.image_to_show, collections='image_summary',
                                       max_outputs=self.params.batchSize)

    def _conv_layer(
            self, layer_name, inputs, filters, size, stride, channels=None, padding='SAME',
            freeze=False, xavier=False, relu=True, stddev=0.001):
        """Convolutional layer operation constructor.

        Args:
          layer_name: layer name.
          inputs: input tensor
          filters: number of output filters.
          size: kernel size.
          stride: stride
          padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
          freeze: if true, then do not train the parameters in this layer.
          xavier: whether to use xavier weight initializer or not.
          relu: whether to use relu or not.
          stddev: standard deviation used for random weight initializer.
        Returns:
          A convolutional layer operation.
        """
        with tf.variable_scope(layer_name) as scope:
            if channels is None:
                channels = inputs.get_shape()[3]
            if xavier:
                kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
                bias_init = tf.constant_initializer(0.0)
            else:
                kernel_init = tf.truncated_normal_initializer(
                    stddev=stddev, dtype=tf.float32)
                bias_init = tf.constant_initializer(0.0)

            kernel = _variable_with_weight_decay(
                'kernels', shape=[size, size, int(channels), filters],
                wd=0.0001, initializer=kernel_init, trainable=(not freeze))

            biases = _variable_on_device('biases', [filters], bias_init,
                                         trainable=(not freeze))
            self.model_params += [kernel, biases]

            conv = tf.nn.conv2d(
                inputs, kernel, [1, stride, stride, 1], padding=padding,
                name='convolution')
            conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')

            if relu:
                out = tf.nn.relu(conv_bias, 'relu')
            else:
                out = conv_bias

            return out

    def _pooling_layer(
            self, layer_name, inputs, size, stride, padding='SAME'):
        """Pooling layer operation constructor.

        Args:
          layer_name: layer name.
          inputs: input tensor
          size: kernel size.
          stride: stride
          padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
        Returns:
          A pooling layer operation.
        """

        with tf.variable_scope(layer_name) as scope:
            out = tf.nn.max_pool(inputs,
                                 ksize=[1, size, size, 1],
                                 strides=[1, stride, stride, 1],
                                 padding=padding)
            return out

    def _fc_layer(
            self, layer_name, inputs, hiddens, flatten=False, relu=True,
            xavier=False, stddev=0.001):
        """Fully connected layer operation constructor.

        Args:
          layer_name: layer name.
          inputs: input tensor
          hiddens: number of (hidden) neurons in this layer.
          flatten: if true, reshape the input 4D tensor of shape
              (batch, height, weight, channel) into a 2D tensor with shape
              (batch, -1). This is used when the input to the fully connected layer
              is output of a convolutional layer.
          relu: whether to use relu or not.
          xavier: whether to use xavier weight initializer or not.
          stddev: standard deviation used for random weight initializer.
        Returns:
          A fully connected layer operation.
        """
        with tf.variable_scope(layer_name) as scope:
            input_shape = inputs.get_shape().as_list()
            if flatten:
                dim = input_shape[1] * input_shape[2] * input_shape[3]
                inputs = tf.reshape(inputs, [-1, dim])
            else:
                dim = input_shape[1]

            if xavier:
                kernel_init = tf.contrib.layers.xavier_initializer()
                bias_init = tf.constant_initializer(0.0)
            else:
                kernel_init = tf.truncated_normal_initializer(
                    stddev=stddev, dtype=tf.float32)
                bias_init = tf.constant_initializer(0.0)

            weights = _variable_with_weight_decay(
                'weights', shape=[dim, hiddens], wd=0.0001,
                initializer=kernel_init)
            biases = _variable_on_device('biases', [hiddens], bias_init)
            self.model_params += [weights, biases]

            outputs = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
            if relu:
                outputs = tf.nn.relu(outputs, 'relu')

            return outputs
