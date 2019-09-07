from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU, CuDNNGRU, CuDNNLSTM
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten, BatchNormalization
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model
from utils import AttentionWithContext



def get_model(maxlen, max_features,embed_size,embedding_matrix,n_classes):
    sequence_input = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(sequence_input)

    x1 = SpatialDropout1D(0.2)(x)

    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x1)

    x = Conv1D(32, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)

    y = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x1)

    y = Conv1D(32, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(y)

    avg_pool1 = GlobalAveragePooling1D()(x)

    max_pool1 = GlobalMaxPooling1D()(x)

    avg_pool2 = GlobalAveragePooling1D()(y)

    max_pool2 = GlobalMaxPooling1D()(y)

    x = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization(x)
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

