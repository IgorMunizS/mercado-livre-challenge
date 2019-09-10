import pickle
from keras.preprocessing import text, sequence

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