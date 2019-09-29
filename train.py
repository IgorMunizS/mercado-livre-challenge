import pandas as pd
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import Adam, Nadam
from sklearn.model_selection import train_test_split
from keras_radam import RAdam
from generator import DataGenerator
from model import get_model, get_small_model, get_three_entrys_model
from utils.tokenizer import tokenize, save_multi_inputs
from utils.embeddings import meta_embedding, CharVectorizer, generated_embedding
from utils.callbacks import Lookahead, CyclicLR
from sklearn.utils import class_weight
import argparse
import sys
import numpy as np
from utils.preprocess import clean_numbers, clean_text, replace_typical_misspell, normalize_title, RemoveStopWords
from utils.features import build_features
from tqdm import tqdm
tqdm.pandas()
from keras.preprocessing import sequence
from utils.utils import label_smooth_loss
from sklearn.feature_extraction.text import HashingVectorizer

def __pretraining(train_new,X_test,max_features,EMBEDDING,embed_size,maxlen,lang,char_vectorizer,type_model,classes,
                 batch_size,char_embed_size):
    X_train = train_new[train_new['label_quality'] == 'reliable']['title']
    Y_train = train_new[train_new['label_quality'] == 'reliable']['category'].values

    # X_hash, X_hash_val, Y_hash, Y_hash_val = train_test_split(X_train, Y_train,
    #                                                           train_size=0.9, random_state=233)

    # X_hash = hash_vec_fitted.transform(X_hash)
    # X_hash_val = hash_vec_fitted.transform(X_hash_val)

    tok, X_train = tokenize(X_train, X_test, max_features, maxlen, lang)
    glove_embedding_matrix = meta_embedding(tok, EMBEDDING[lang][0], max_features, embed_size, lang)
    fast_embedding_matrix = meta_embedding(tok, EMBEDDING[lang][1], max_features, embed_size, lang)

    char_embedding = char_vectorizer.get_char_embedding(tok)

    # embedding_matrix = np.mean([glove_embedding_matrix, fast_embedding_matrix], axis=0)

    embedding_matrix = np.concatenate((glove_embedding_matrix, fast_embedding_matrix, char_embedding), axis=1)

    if type_model == 'three':
        # X_train_2 = train_new[train_new['label_quality'] == 'reliable']['small_title']
        X_train_3 = train_new[train_new['label_quality'] == 'reliable'][train_new.columns[6:]].values

        # X_train_2 = tok.texts_to_sequences(X_train_2)
        # X_train_2 = sequence.pad_sequences(X_train_2, maxlen=6)

        X_train, X_val, X_train_3, X_val_3, Y_train, Y_val = train_test_split(X_train, X_train_3, Y_train,
                                                                              train_size=0.9, random_state=233)

        train_generator = DataGenerator([X_train, X_train_3], Y_train, classes, batch_size=batch_size, mode=type_model,
                                        resample=False)
        val_generator = DataGenerator([X_val, X_val_3], Y_val, classes, batch_size=batch_size, mode=type_model,
                                      resample=False)

    else:

        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.9, random_state=233)

        train_generator = DataGenerator(X_train, Y_train, classes, batch_size=batch_size, resample=False)
        val_generator = DataGenerator(X_val, Y_val, classes, batch_size=batch_size, resample=False)

    opt = Adam(lr=1e-3)
    # opt = Nadam(lr=1e-3, schedule_decay=0.005)
    # opt = Adam(lr=1e-3)
    if type_model == 'small':
        model = get_small_model(maxlen, max_features, 2 * embed_size + char_embed_size, embedding_matrix, len(classes))

    elif type_model == 'three':
        model = get_three_entrys_model(maxlen, max_features, 2 * embed_size + char_embed_size, embedding_matrix,
                                       len(classes))

    else:
        model = get_model(maxlen, max_features, 2 * embed_size + char_embed_size, embedding_matrix, len(classes))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    print("Pré treinando")

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=1,
        verbose=1,
        mode='auto',
        epsilon=0.0001,
        cooldown=0,
        min_lr=0
    )
    early = EarlyStopping(monitor="val_loss", mode="min", patience=3)

    callbacks_list = [early, reduce_lr]

    model.fit_generator(generator=train_generator,
                        validation_data=val_generator,
                        callbacks=callbacks_list,
                        epochs=30,
                        use_multiprocessing=True,
                        workers=42)

    return model


def __training(train_new,X_test,max_features,maxlen,lang,EMBEDDING,embed_size,char_vectorizer, char_embed_size,classes,type_model,test_new,
               batch_size,model=None):
    X_train = train_new['title']

    Y_train = train_new['category'].values

    tok, X_train = tokenize(X_train, X_test, max_features, maxlen, lang)

    word_index = tok.word_index
    # prepare embedding matrix
    max_features = min(max_features, len(word_index) + 1)
    # max_features = len(word_index)
    glove_embedding_matrix = meta_embedding(tok, EMBEDDING[lang][0], max_features, embed_size, lang)
    fast_embedding_matrix = meta_embedding(tok, EMBEDDING[lang][1], max_features, embed_size, lang)
    generated_fast_embedding_matrix = generated_embedding(tok,max_features,embed_size,lang)

    # char_embedding = char_vectorizer.get_char_embedding(tok)

    # embedding_matrix = np.mean([glove_embedding_matrix, fast_embedding_matrix], axis=0)

    embedding_matrix = np.concatenate((generated_fast_embedding_matrix, fast_embedding_matrix, glove_embedding_matrix), axis=1)

    # embedding_matrix = generated_fast_embedding_matrix

    # class_weights = class_weight.compute_class_weight('balanced',
    #                                                   classes,
    #                                                   Y_train)

    if model is None:
        if type_model == 'small':
            model = get_small_model(maxlen, max_features, 2 * embed_size + char_embed_size, embedding_matrix,
                                    len(classes))

        elif type_model == 'three':
            model = get_three_entrys_model(maxlen, max_features, 3 * embed_size, embedding_matrix,
                                           len(classes))

        else:
            model = get_model(maxlen, max_features, 2 * embed_size + char_embed_size, embedding_matrix, len(classes))


    if type_model == 'three':
        # X_train_2 = train_new['small_title']
        X_train_3 = train_new[train_new.columns[6:]].values

        # X_train_2 = tok.texts_to_sequences(X_train_2)
        # X_train_2 = sequence.pad_sequences(X_train_2, maxlen=6)

        X_test_small = test_new["small_title"]
        X_test_small = tok.texts_to_sequences(X_test_small)
        X_test_small = sequence.pad_sequences(X_test_small, maxlen=6)
        X_test_features = test_new[test_new.columns[5:]].values

        save_multi_inputs(X_test_small, X_test_features, lang)

        X_train, X_val, X_train_3, X_val_3, Y_train, Y_val = train_test_split(X_train, X_train_3, Y_train,
                                                                              train_size=0.9, random_state=233)

        train_generator = DataGenerator([X_train, X_train_3], Y_train, classes, batch_size=batch_size, mode=type_model,
                                        resample=True)
        val_generator = DataGenerator([X_val, X_val_3], Y_val, classes, batch_size=batch_size, mode=type_model,
                                      resample=False)
        model.get_layer('embedding_layer').set_weights([embedding_matrix])
        # model.get_layer('small_embedding_layer').set_weights([embedding_matrix])

    else:

        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.9, random_state=233,
                                                          stratify=Y_train)

        train_generator = DataGenerator(X_train, Y_train, classes, batch_size=batch_size, resample=True)
        val_generator = DataGenerator(X_val, Y_val, classes, batch_size=batch_size, resample=False)

        model.layers[1].set_weights([embedding_matrix])



    opt = RAdam(lr=0.001)

    model.compile(loss=label_smooth_loss, optimizer=opt, metrics=['accuracy'])


    filepath = '../models/' + lang + '_model_{epoch:02d}_{val_acc:.4f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max',
                                 save_weights_only=False)
    early = EarlyStopping(monitor="val_acc", mode="max", patience=3)

    clr = CyclicLR(base_lr=0.0003, max_lr=0.001,
                   step_size=35000, reduce_on_plateau=1, monitor='val_loss', reduce_factor=10)

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=1,
        verbose=1,
        mode='auto',
        epsilon=0.0001,
        cooldown=0,
        min_lr=0
    )

    callbacks_list = [checkpoint, early, reduce_lr]

    lookahead = Lookahead(k=5, alpha=0.5)  # Initialize Lookahead
    lookahead.inject(model)

    print("Treinando")

    model.fit_generator(generator=train_generator,
                        validation_data=val_generator,
                        callbacks=callbacks_list,
                        epochs=50,
                        use_multiprocessing=True,
                        workers=42)



def training(languages, EMBEDDING,train,test,type_model,pre):

    for lang in languages:
        print('Training ',lang)
        train_new = train[train["language"] == lang]
        test_new = test[test["language"] == lang]

        train_new['title'] = train_new['title'].str.lower()
        test_new['title'] = test_new['title'].str.lower()

        if type_model == 'three':
            train_new = build_features(train_new)
            test_new = build_features(test_new)

        stopwords = RemoveStopWords(lang)

        train_new["title"] = train_new["title"].progress_apply(lambda x: replace_typical_misspell(x, lang))
        # train_new["title"] = train_new["title"].progress_apply(lambda x: clean_numbers(x))
        # train_new["title"] = train_new["title"].progress_apply(lambda x: stopwords.remove_stopwords(x, lang))
        train_new["title"] = train_new["title"].progress_apply(lambda x: clean_text(x))
        train_new["title"] = train_new["title"].progress_apply(lambda x: normalize_title(x))

        test_new["title"] = test_new["title"].progress_apply(lambda x: replace_typical_misspell(x, lang))
        # test_new["title"] = test_new["title"].progress_apply(lambda x: clean_numbers(x))
        # test_new["title"] = test_new["title"].progress_apply(lambda x: stopwords.remove_stopwords(x, lang))
        test_new["title"] = test_new["title"].progress_apply(lambda x: clean_text(x))
        test_new["title"] = test_new["title"].progress_apply(lambda x: normalize_title(x))

        classes = train_new["category"].unique()

        X_test = test_new["title"]

        max_features = 300000
        maxlen = 20
        embed_size = 300
        batch_size = 512

        # Generate char embedding without preprocess
        text = (train_new['title'].tolist() + test_new["title"].tolist())
        char_vectorizer = CharVectorizer(max_features,text)
        char_embed_size = char_vectorizer.embed_size

        #
        # hash_vectorizer = HashingVectorizer(n_features=max_features)
        # hash_vec_fitted = hash_vectorizer.fit(text)



        if pre:

            pre_model = __pretraining(train_new,X_test,max_features,EMBEDDING,embed_size,maxlen,lang,char_vectorizer,
                                 type_model,classes,batch_size,char_embed_size)

            __training(train_new,X_test,max_features,maxlen,lang,EMBEDDING,embed_size,char_vectorizer, char_embed_size,classes,type_model,test_new,
               batch_size,model=pre_model)

        else:
            __training(train_new,X_test,max_features,maxlen,lang,EMBEDDING,embed_size,char_vectorizer, char_embed_size,classes,type_model,test_new,
               batch_size,model=None)


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Predict script')


    parser.add_argument('--model', help='Local of training', default='normal')
    parser.add_argument('--pre', help='Pretraining with only reliable values', default=False, type=bool)
    parser.add_argument('--small_set', default=False, type=bool)
    parser.add_argument('--language', help='Training only in a specific language', default='both')
    parser.add_argument('--data_folder', default='../../dados/')
    parser.add_argument('--embedding_folder', default='../../../harold/word_embeddings/')



    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)

    train = pd.read_csv(args.data_folder + "train.csv")
    test = pd.read_csv(args.data_folder + "test.csv")



    EMBEDDING = {"spanish": [args.embedding_folder + "espanhol/glove-sbwc.i25.vec",
                             args.embedding_folder + "espanhol/fasttext-nlarge-suc.vec"],

                 "portuguese": [args.embedding_folder + "portugues/glove_s300.txt",
                                args.embedding_folder + "portugues/skip_s300.txt"]}

    if args.language == 'both':
        languages = ['portuguese', 'spanish']
    else:
        languages = []
        languages.append(args.language)

    if args.small_set:
        _, train = train_test_split(train, test_size=int(.1 * len(train)), random_state=42, stratify=train.category)

    training(languages,EMBEDDING,train,test,args.model, args.pre)
