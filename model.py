from keras.layers import Dense,Input, Bidirectional, Conv1D, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D, Dropout
from keras.models import Model
from utils.layers import AttentionWithContext
from utils.embeddings import DynamicMetaEmbedding
from keras import Sequential
import tensorflow as tf


def get_model(maxlen, max_features,embed_size,embedding_matrix,n_classes):
    sequence_input = Input(shape=(maxlen,))

    # fast_embedding = tf.keras.layers.Embedding(max_features, embed_size,
    #                                           embeddings_initializer=tf.keras.initializers.Constant(fast_embedding_matrix),
    #                                           trainable=False)
    # glove_embedding = tf.keras.layers.Embedding(max_features,
    #                                             embed_size,
    #                                             embeddings_initializer=tf.keras.initializers.Constant(glove_embedding_matrix),
    #                                             trainable=False)
    #
    # embedding_model = tf.keras.Sequential([tf.keras.layers.Input(shape=(maxlen,), dtype='int32'),
    #                              DynamicMetaEmbedding([fast_embedding, glove_embedding])])

    embedding = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(sequence_input)

    # x = DynamicMetaEmbedding([fast_embedding, glove_embedding])()

    x = SpatialDropout1D(0.3)(embedding)
    x1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)
    x2 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x1)
    x3 = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x2)
    max_pool1 = GlobalMaxPooling1D()(x1)
    max_pool2 = GlobalMaxPooling1D()(x2)
    max_pool3 = GlobalMaxPooling1D()(x3)
    x = concatenate([max_pool1,max_pool2,max_pool3])

    # x1 = SpatialDropout1D(0.2)(x)
    #
    # x = Bidirectional(CuDNNGRU(256, return_sequences=True))(embedding)
    #
    # x = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)
    #
    # x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    #
    # x = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)
    #
    # y = Bidirectional(CuDNNLSTM(256, return_sequences=True))(embedding)
    #
    # y = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(y)
    #
    # y = Bidirectional(CuDNNLSTM(128, return_sequences=True))(y)
    #
    # y = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(y)
    #
    # avg_pool1 = GlobalAveragePooling1D()(x)
    #
    # max_pool1 = GlobalMaxPooling1D()(x)
    #
    # avg_pool2 = GlobalAveragePooling1D()(y)
    #
    # max_pool2 = GlobalMaxPooling1D()(y)
    #
    # x = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])

    preds = Dense(n_classes, activation="softmax")(x)
    model = Model(sequence_input, preds)

    return model

def get_three_entrys_model(maxlen, max_features,embed_size,embedding_matrix,n_classes):
    sequence_input = Input(shape=(maxlen,))
    # small_sequence_input = Input(shape=(6,))
    features_input = Input(shape=(20,))

    embedding_1 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True, name='embedding_layer')(sequence_input)

    x = SpatialDropout1D(0.3)(embedding_1)
    x1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)
    x2 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x1)
    x3 = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x2)
    x4 = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x1)

    max_pool1 = GlobalMaxPooling1D()(x1)
    max_pool2 = GlobalMaxPooling1D()(x2)
    max_pool3 = GlobalMaxPooling1D()(x3)
    max_pool4 = GlobalMaxPooling1D()(x4)

    # x1 = SpatialDropout1D(0.3)(embedding_1)
    #
    # x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x1)
    #
    # x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    #
    # x = AttentionWithContext()(x)
    # dense_attention = Dense(64, activation="relu")(x)

    # average_pool_attention = GlobalAveragePooling1D()(x)


    # x1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    # x = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x1)
    # max_pool1 = GlobalMaxPooling1D()(x)


    # embedding_2 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False,
    #                         name='small_embedding_layer')(small_sequence_input)
    #
    x = SpatialDropout1D(0.3)(embedding_1)

    # x1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    # x = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x1)
    # max_pool2 = GlobalMaxPooling1D()(x)

    x1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)
    x2 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x1)
    x3 = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x2)


    avg_pool4 = GlobalAveragePooling1D()(x1)
    avg_pool5 = GlobalAveragePooling1D()(x2)
    max_pool6 = GlobalMaxPooling1D()(x2)
    max_pool7 = GlobalMaxPooling1D()(x3)

    x_concat = concatenate([avg_pool4,avg_pool5,max_pool6,max_pool7])
    dense_1 = Dense(768, activation='relu')(x_concat)
    dense_2 = Dense(768, activation='relu')(x_concat)

    x_concat_2 = concatenate([x_concat,dense_1,dense_2])

    features_dense = Dense(768, activation="relu")(features_input)

    x = concatenate([max_pool1,max_pool2,max_pool3,max_pool4,x_concat_2,features_dense])

    # x = concatenate([max_pool1, max_pool2,features_dense])
    # x = Dense(128, activation='relu')(concat)
    # x = Dropout(0.1)(x)
    # x = BatchNormalization()(x)
    #
    # x = concatenate([concat, x])

    preds = Dense(n_classes, activation="softmax")(x)
    model = Model(inputs=[sequence_input, features_input], outputs=preds)

    return model

def get_small_model(maxlen, max_features,embed_size,embedding_matrix,n_classes):
    sequence_input = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(sequence_input)

    x1 = SpatialDropout1D(0.3)(x)

    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x1)

    x = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)

    max_pool1 = GlobalMaxPooling1D()(x)

    # x = concatenate([avg_pool1, max_pool1])

    preds = Dense(n_classes, activation="softmax")(max_pool1)
    model = Model(sequence_input, preds)

    return model

def get_attention_model(maxlen, max_features,embed_size,embedding_matrix,n_classes):
    sequence_input = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(sequence_input)

    x1 = SpatialDropout1D(0.2)(x)

    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x1)

    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)

    x = AttentionWithContext()(x)
    x = Dense(64, activation="relu")(x)

    preds = Dense(n_classes, activation="softmax")(x)
    model = Model(sequence_input, preds)

    return model

