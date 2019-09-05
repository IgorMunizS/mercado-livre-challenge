import pickle
from keras.preprocessing import text, sequence
import tensorflow as tf
import keras.backend as K
import numpy as np

def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def tokenize(X_train,X_test,max_features, maxlen, lang):
    print("Tokenizando")
    tok = text.Tokenizer(num_words=max_features, lower=True)
    tok.fit_on_texts(list(X_train) + list(X_test))
    X_train = tok.texts_to_sequences(X_train)
    X_test = tok.texts_to_sequences(X_test)
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    # saving
    with open('../tokenizers/' + lang + '_tokenizer.pickle', 'wb') as handle:
        pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tok, X_train


def embedding(tok,embedding_file,max_features,embed_size):
    print("Gerando Embedding")


    embeddings_index = {}
    with open(embedding_file,encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    word_index = tok.word_index
    #prepare embedding matrix
    num_words = min(max_features, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix
