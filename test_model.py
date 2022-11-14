
import numpy as np

import tensorflow as tf

print("current ver is ", tf.__version__)
print("np version is ",np.__version__)


class SENetLayer(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=3, seed=1024, **kwargs):
        super(SENetLayer, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.seed = seed

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('`SENetLayer` layer should be called \
                on a list of at least 2 inputs')

        self.field_size = len(input_shape)
        self.embedding_size = input_shape[0].as_list()[-1]
        reduction_size = max(1, int(self.field_size // self.reduction_ratio))

        # 定义两个全连接层
        self.W_1 = self.add_weight(name="W_1", shape=(self.field_size,
                                                      reduction_size), initializer=glorot_normal(self.seed))
        self.W_2 = self.add_weight(name="W_2", shape=(reduction_size,
                                                      self.field_size), initializer=glorot_normal(self.seed))
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        """inputs: 是一个长度为field_size的列表，其中每个元素的形状为：
            (None, 1, embedding_size)
        """
        inputs = tf.keras.layers.Concatenate(axis=1)(inputs)  # (None, field_size, embedding_size)
        x = tf.reduce_mean(inputs, axis=-1)  # (None, field_size)

        # (None, field_size) * (field_size, reduction_size) =
        # (None, reduction_size)
        A_1 = tf.tensordot(x, self.W_1, axes=(-1, 0))
        A_1 = tf.nn.relu(A_1)
        # (None, reduction_size) * (reduction_size, field_size) =
        # (None, field_size)
        A_2 = tf.tensordot(A_1, self.W_2, axes=(-1, 0))
        A_2 = tf.nn.relu(A_2)
        A_2 = tf.expand_dims(A_2, axis=2)  # (None, field_size, 1)

        res = tf.multiply(inputs, A_2)  # (None, field_size, embedding_size)
        # 切分成数组，方便后续特征交叉
        res = tf.split(res, self.field_size, axis=1)
        return res

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"reduction_ratio": self.reduction_ratio, "seed": self.seed}
        base_config = super(SENetLayer, self).get_config()
        base_config.update(config)
        return base_config

class SENETLayer(tf.keras.layers.Layer):
    """SENETLayer used in FiBiNET.
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Output shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Arguments
        - **reduction_ratio** : Positive integer, dimensionality of the
         attention network output space.
        - **seed** : A Python integer to use as random seed.
      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)
    """

    def __init__(self, reduction_ratio=3, seed=1024, **kwargs):
        self.reduction_ratio = reduction_ratio

        self.seed = seed
        super(SENETLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `AttentionalFM` layer should be called '
                             'on a list of at least 2 inputs')

        self.filed_size = len(input_shape)
        self.embedding_size = input_shape[0][-1]
        reduction_size = max(1, self.filed_size // self.reduction_ratio)

        self.W_1 = self.add_weight(shape=(
            self.filed_size, reduction_size), initializer=glorot_normal(seed=self.seed), name="W_1")
        self.W_2 = self.add_weight(shape=(
            reduction_size, self.filed_size), initializer=glorot_normal(seed=self.seed), name="W_2")

        self.tensordot = Lambda(
            lambda x: tf.tensordot(x[0], x[1], axes=(-1, 0)))

        # Be sure to call this somewhere!
        super(SENETLayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):

        if K.ndim(inputs[0]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        inputs = concat_func(inputs, axis=1)
        Z = reduce_mean(inputs, axis=-1, )

        A_1 = tf.nn.relu(self.tensordot([Z, self.W_1]))
        A_2 = tf.nn.relu(self.tensordot([A_1, self.W_2]))
        V = tf.multiply(inputs, tf.expand_dims(A_2, axis=2))

        return tf.split(V, self.filed_size, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return [None] * self.filed_size

    @property
    def get_config(self, ):
        config = {'reduction_ratio': self.reduction_ratio, 'seed': self.seed}
        base_config = super(SENETLayer, self).get_config()
        base_config.update(config)
        return base_config

