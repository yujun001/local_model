
import tensorflow as tf

class Interaction(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Interaction, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Interaction, self).build(input_shape)

    def call(self, x, **kwargs):
        sum_square = tf.square(tf.reduce_sum(x, axis=1))
        square_sum = tf.reduce_sum(tf.square(x), axis=1)
        bi_interact = tf.multiply(0.5, tf.subtract(sum_square, square_sum))
        return bi_interact

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        config = super(Interaction, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class EmbeddingLookup(tf.keras.layers.Layer):
    def __init__(self, embedding, **kwargs):
        super(EmbeddingLookup, self).__init__(**kwargs)
        self.embedding = embedding

    def build(self, input_shape):
        super(EmbeddingLookup, self).build(input_shape)

    def call(self, inputs, **kwargs):
        idx = inputs
        embed = tf.nn.embedding_lookup(params=self.embedding, ids=idx)
        return embed

    def get_config(self):
        config = super(EmbeddingLookup, self).get_config()
        return config


class DenseToSparseTensor(tf.keras.layers.Layer):
    def __init__(self, mask_value=-1):
        super(DenseToSparseTensor, self).__init__()
        self.mask_value = mask_value

    def call(self, dense_tensor, **kwargs):
        idx = tf.where(tf.not_equal(dense_tensor, tf.constant(self.mask_value, dtype=dense_tensor.dtype)))
        sparse_tensor = tf.SparseTensor(idx, tf.gather_nd(dense_tensor, idx), tf.shape(dense_tensor, out_type=tf.int64))
        return sparse_tensor

    def get_config(self):
        config = super(DenseToSparseTensor, self).get_config()
        config.update({'mask_value': self.mask_value})
        return config

class EmbeddingLookupSparse(tf.keras.layers.Layer):
    def __init__(self, embedding, has_weight=False, combiner='sum', **kwargs):
        super(EmbeddingLookupSparse, self).__init__(**kwargs)
        self.has_weight = has_weight
        self.combiner = combiner
        self.embedding = embedding

    def build(self, input_shape):
        super(EmbeddingLookupSparse, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.has_weight:
            idx, val = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding, sp_ids=idx, sp_weights=val,
                                                           combiner=self.combiner)
        else:
            idx = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding, sp_ids=idx, sp_weights=None,
                                                           combiner=self.combiner)
        return tf.expand_dims(combiner_embed, 1)

    def get_config(self):
        config = super(EmbeddingLookupSparse, self).get_config()
        config.update({'has_weight': self.has_weight, 'combiner': self.combiner})
        return config


def build_model():

    # input
    # (1) user feature
    # 1.1 sparse
    input_unique_int = tf.keras.layers.Input(shape=(1,), name='unique_int', dtype='int32')
    input_user_country = tf.keras.layers.Input(shape=(1,), name='user_country', dtype='int32')
    input_model = tf.keras.layers.Input(shape=(1,), name='model', dtype='int32')

    # 1.2 dense
    input_u30dvv = tf.keras.layers.Input(shape=(1,), name='u30dvv', dtype='float32')
    input_u7dvv = tf.keras.layers.Input(shape=(1,), name='u7dvv', dtype='float32')
    input_u7dy7s = tf.keras.layers.Input(shape=(1,), name='u7dy7s', dtype='float32')

    # (2) video feature
    # 2.1 sparse
    input_video_id = tf.keras.layers.Input(shape=(1,), name='video_id', dtype='int32')
    input_tag = tf.keras.layers.Input(shape=(1,), name='tag', dtype='int32')
    input_category = tf.keras.layers.Input(shape=(1,), name='category', dtype='int32')
    input_video_country = tf.keras.layers.Input(shape=(1,), name='video_country', dtype='int32')

    # 2.2 dense
    input_exp_num = tf.keras.layers.Input(shape=(1,), name='iexpnum', dtype='float32')
    input_view_num = tf.keras.layers.Input(shape=(1,), name='iviewnum', dtype='float32')
    input_play_num = tf.keras.layers.Input(shape=(1,), name='iplaynum', dtype='float32')
    input_share_num = tf.keras.layers.Input(shape=(1,), name='isharenum', dtype='float32')
    input_effplayratio_num = tf.keras.layers.Input(shape=(1,), name='ieffplayratio', dtype='float32')

    # (5) 无权重序列
    input_u15stagseq = tf.keras.layers.Input(shape=(None,), name='u15stagseq', dtype='int32')
    input_u15svidseq = tf.keras.layers.Input(shape=(None,), name='u15svidseq', dtype='int32')

    # (6) 有权重序列
    input_u7dtagfavorseq = tf.keras.layers.Input(shape=(None,), name='u7dtagfavorseq', dtype='int32')
    input_u7dtagfavorseq_weight = tf.keras.layers.Input(shape=(None,), name='u7dtagfavorseq_weight', dtype='float32')

    # embedding
    # ----------------------------------------------------------------------------------------------------------------
    # sparse
    sparse_unique_int_matrix = tf.Variable(initial_value=tf.random.truncated_normal(shape=(10000000+2,2)), trainable=True, name='sparse_unique_int_matrix')
    sparse_video_id_int_matrix = tf.Variable(initial_value=tf.random.truncated_normal(shape=(10000000+2,2)), trainable=True, name='sparse_video_id_int_matrix')
    sparse_tag_matrix = tf.Variable(initial_value=tf.random.truncated_normal(shape=(100 + 2, 2)), trainable=True, name='sparse_tag_matrix')
    sparse_category_matrix = tf.Variable(initial_value=tf.random.truncated_normal(shape=(100 + 2, 2)), trainable=True, name='sparse_category_matrix')
    sparse_country_matrix = tf.Variable(initial_value=tf.random.truncated_normal(shape=(100 + 2, 2)), trainable=True, name='sparse_country_matrix')
    sparse_model_matrix = tf.Variable(initial_value=tf.random.truncated_normal(shape=(100 + 2, 2)), trainable=True, name='sparse_model_matrix')

    sparse_unique_int_embedding = EmbeddingLookup(embedding=sparse_unique_int_matrix, name='kd_emb_unique_int')(input_unique_int)
    sparse_user_country_embedding = EmbeddingLookup(embedding=sparse_country_matrix, name='kd_emb_country')(input_user_country)
    sparse_model_embedding = EmbeddingLookup(embedding=sparse_model_matrix, name='kd_emb_model')(input_model)

    sparse_video_int_embedding = EmbeddingLookup(embedding=sparse_video_id_int_matrix, name='kd_emb_video_id_int')(input_video_id)
    sparse_tag_embedding = EmbeddingLookup(embedding=sparse_tag_matrix, name='kd_emb_tag')(input_tag)
    sparse_category_embedding = EmbeddingLookup(embedding=sparse_category_matrix, name='kd_emb_category')(input_category)
    sparse_video_country_embedding = EmbeddingLookup(embedding=sparse_country_matrix, name='kd_emb_model')(input_video_country)

    # sparse seq no weight
    sparse_u15stagseq_kd_embedding = EmbeddingLookup(embedding=sparse_tag_matrix, name='kd_emb_u15stagseq')(input_u15stagseq)
    sparse_u15svidseq_kd_embedding = EmbeddingLookup(embedding=sparse_video_id_int_matrix, name='kd_emb_u15svidseq')(input_u15svidseq)

    # sparse seq has weight
    id_seq_u7dtagfavorseq = DenseToSparseTensor()(input_u7dtagfavorseq)
    weight_seq_u7dtagfavorseq = DenseToSparseTensor()(input_u7dtagfavorseq_weight)
    sparse_u7dtagfavorseq_embedding = EmbeddingLookupSparse(embedding=sparse_tag_matrix, has_weight=True)([id_seq_u7dtagfavorseq, weight_seq_u7dtagfavorseq])

    # ----------------------------------------------------------------------------------------------------------------
    # user part
    user_dense_embedding_list = [
        input_u30dvv,
        input_u7dvv,
        input_u7dy7s]
    user_dense_embedding = [tf.expand_dims(user_dense_embedding_layer, axis=-2) for user_dense_embedding_layer in user_dense_embedding_list]
    user_sparse_embedding_list = [sparse_unique_int_embedding,
                                  sparse_user_country_embedding,
                                  sparse_model_embedding,
                                  sparse_u15stagseq_kd_embedding,
                                  sparse_u15svidseq_kd_embedding,
                                  sparse_u7dtagfavorseq_embedding
                                  ]
    user_embedding_list = user_dense_embedding + user_sparse_embedding_list

    # video part
    video_dense_embedding_list = [
        input_exp_num,
        input_view_num,
        input_play_num,
        input_share_num,
        input_effplayratio_num]
    video_dense_embedding = [tf.expand_dims(video_dense_embedding_layer, axis=-2) for video_dense_embedding_layer in video_dense_embedding_list]
    video_sparse_embedding_list = [sparse_video_int_embedding,
                                  sparse_tag_embedding,
                                  sparse_category_embedding,
                                  sparse_video_country_embedding
                                  ]
    video_embedding_list = video_dense_embedding + video_sparse_embedding_list

    # ----------------------------------------------------------------------------------------------------------------
    # user mlp
    user_embedding = tf.keras.layers.Concatenate()(user_embedding_list)
    user_embedding = tf.keras.layers.BatchNormalization()(user_embedding)
    bi_user_embedding = Interaction()(user_embedding)
    user_dense_tensor = tf.keras.layers.Flatten()(user_embedding)

    user_dense_tensor = tf.keras.layers.Dropout(0.5)(user_dense_tensor)
    user_dense_tensor = tf.keras.layers.Dense(512, activation=tf.nn.relu)(user_dense_tensor)

    user_dense_tensor = tf.concat([user_dense_tensor, bi_user_embedding], axis=-1)
    user_dense_tensor = tf.keras.layers.BatchNormalization()(user_dense_tensor)
    user_dense_tensor = tf.keras.layers.Dropout(0.5)(user_dense_tensor)
    user_dense_tensor = tf.keras.layers.Dense(256, activation=tf.nn.relu)(user_dense_tensor)

    user_dense_tensor = tf.keras.layers.BatchNormalization()(user_dense_tensor)
    user_dense_tensor = tf.keras.layers.Dropout(0.5)(user_dense_tensor)
    user_dense_tensor = tf.keras.layers.Dense(128, name='user_embedding')(user_dense_tensor)
    user_dense_tensor = tf.nn.l2_normalize(user_dense_tensor, axis=1)

    # ----------------------------------------------------------------------------------------------------------------
    # video mlp_1
    video_embedding_1 = tf.keras.layers.Concatenate()(video_embedding_list)
    video_embedding_1 = tf.keras.layers.BatchNormalization()(video_embedding_1)
    bi_video_embedding_1 = Interaction()(video_embedding_1)
    video_dense_tensor_1 = tf.keras.layers.Flatten()(video_embedding_1)

    video_dense_tensor_1 = tf.keras.layers.Dropout(0.5)(video_dense_tensor_1)
    video_dense_tensor_1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(video_dense_tensor_1)

    video_dense_tensor_1 = tf.concat([video_dense_tensor_1, bi_video_embedding_1], axis=-1)
    video_dense_tensor_1 = tf.keras.layers.BatchNormalization()(video_dense_tensor_1)
    video_dense_tensor_1 = tf.keras.layers.Dropout(0.5)(video_dense_tensor_1)
    video_dense_tensor_1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(video_dense_tensor_1)

    video_dense_tensor_1 = tf.keras.layers.BatchNormalization()(video_dense_tensor_1)
    video_dense_tensor_1 = tf.keras.layers.Dropout(0.5)(video_dense_tensor_1)
    video_dense_tensor_1 = tf.keras.layers.Dense(128, name='video_embedding_1')(video_dense_tensor_1)
    video_dense_tensor_1 = tf.nn.l2_normalize(video_dense_tensor_1, axis=1)

    # ----------------------------------------------------------------------------------------------------------------
    # video mlp_2
    video_embedding_2 = tf.keras.layers.Concatenate()(video_embedding_list)
    video_embedding_2 = tf.keras.layers.BatchNormalization()(video_embedding_2)
    bi_video_embedding_2 = Interaction()(video_embedding_2)
    video_dense_tensor_2 = tf.keras.layers.Flatten()(video_embedding_2)

    video_dense_tensor_2 = tf.keras.layers.Dropout(0.5)(video_dense_tensor_2)
    video_dense_tensor_2 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(video_dense_tensor_2)

    video_dense_tensor_2 = tf.concat([video_dense_tensor_2, bi_video_embedding_2], axis=-1)
    video_dense_tensor_2 = tf.keras.layers.BatchNormalization()(video_dense_tensor_2)
    video_dense_tensor_2 = tf.keras.layers.Dropout(0.5)(video_dense_tensor_2)
    video_dense_tensor_2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(video_dense_tensor_2)

    video_dense_tensor_2 = tf.keras.layers.BatchNormalization()(video_dense_tensor_2)
    video_dense_tensor_2 = tf.keras.layers.Dropout(0.5)(video_dense_tensor_2)
    video_dense_tensor_2 = tf.keras.layers.Dense(128, name='video_embedding_1')(video_dense_tensor_2)
    video_dense_tensor_2 = tf.nn.l2_normalize(video_dense_tensor_2, axis=1)

    # ----------------------------------------------------------------------------------------------------------------
    ctr_score = tf.keras.layers.Dot(axes=1)([user_dense_tensor, video_dense_tensor_1])
    ctr_out = tf.clip_by_value(ctr_score, 0, 1, name='ctr_out')

    view_15s_score = tf.keras.layers.Dot(axes=1)([user_dense_tensor, video_dense_tensor_2])
    view_15s_out = tf.clip_by_value(view_15s_score, 0, 1, name='view_15s_out')

    # output
    tower_out = tf.multiply(ctr_out, view_15s_out, name='tower_out')

    user_inputs_list=[
        input_unique_int,
        input_user_country,
        input_model,
        input_u30dvv,
        input_u7dvv,
        input_u7dy7s,
        input_u15stagseq,
        input_u15svidseq,
    ]

    video_inputs_list=[
        input_video_id,
        input_tag,
        input_category,
        input_video_country,
        input_exp_num,
        input_view_num,
        input_play_num,
        input_share_num,
        input_effplayratio_num,
    ]

    model = tf.keras.Model(inputs=user_inputs_list + video_inputs_list,
                           outputs=[ctr_out, tower_out])
    return model