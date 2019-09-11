from keras.layers import Dense,Input, Bidirectional, Conv1D, CuDNNGRU, CuDNNLSTM
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
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

    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(sequence_input)

    # x = DynamicMetaEmbedding([fast_embedding, glove_embedding])()

    x1 = SpatialDropout1D(0.2)(x)

    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x1)

    x = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)

    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)

    x = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)

    y = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x1)

    y = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(y)

    y = Bidirectional(CuDNNLSTM(128, return_sequences=True))(y)

    y = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(y)

    avg_pool1 = GlobalAveragePooling1D()(x)

    max_pool1 = GlobalMaxPooling1D()(x)

    avg_pool2 = GlobalAveragePooling1D()(y)

    max_pool2 = GlobalMaxPooling1D()(y)

    x = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])
    preds = Dense(n_classes, activation="softmax")(x)
    model = Model(sequence_input, preds)

    return model

def get_small_model(maxlen, max_features,embed_size,embedding_matrix,n_classes):
    sequence_input = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(sequence_input)

    x1 = SpatialDropout1D(0.2)(x)

    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x1)

    x = Conv1D(32, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)

    avg_pool1 = GlobalAveragePooling1D()(x)

    max_pool1 = GlobalMaxPooling1D()(x)

    x = concatenate([avg_pool1, max_pool1])
    preds = Dense(n_classes, activation="softmax")(x)
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

