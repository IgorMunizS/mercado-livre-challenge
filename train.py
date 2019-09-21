import pandas as pd
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras_radam import RAdam
from generator import DataGenerator
from model import get_model, get_small_model, get_three_entrys_model
from utils.tokenizer import tokenize, save_multi_inputs
from utils.embeddings import meta_embedding, CharVectorizer
from utils.callbacks import Lookahead, CyclicLR
from sklearn.utils import class_weight
import argparse
import sys
import numpy as np
from utils.preprocess import clean_numbers, clean_text, replace_typical_misspell, normalize_title
from utils.features import build_features
from tqdm import tqdm
tqdm.pandas()
from keras.preprocessing import sequence
from utils.utils import label_smooth_loss
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

        train_new["title"] = train_new["title"].progress_apply(lambda x: clean_numbers(x))
        train_new["title"] = train_new["title"].progress_apply(lambda x: replace_typical_misspell(x, lang))
        train_new["title"] = train_new["title"].progress_apply(lambda x: clean_text(x))
        train_new["title"] = train_new["title"].progress_apply(lambda x: normalize_title(x))

        test_new["title"] = test_new["title"].progress_apply(lambda x: clean_numbers(x))
        test_new["title"] = test_new["title"].progress_apply(lambda x: replace_typical_misspell(x, lang))
        test_new["title"] = test_new["title"].progress_apply(lambda x: clean_text(x))
        test_new["title"] = test_new["title"].progress_apply(lambda x: normalize_title(x))



        X_train = train_new['title']

        Y_train = train_new['category'].values

        classes = train_new["category"].unique()

        X_test = test_new["title"]

        max_features = 100000
        maxlen = 20
        embed_size = 300
        batch_size = 512

        # Generate char embedding without preprocess
        # text = (train_new['title'].tolist() + test_new["title"].tolist())

        # char_vectorizer = CharVectorizer(max_features,text)
        # char_embed_size = char_vectorizer.embed_size


        if pre:
            X_train = train_new[train_new['label_quality'] == 'reliable']['title']
            Y_train = train_new[train_new['label_quality'] == 'reliable']['category'].values


            tok, X_train = tokenize(X_train, X_test, max_features, maxlen, lang)
            glove_embedding_matrix = meta_embedding(tok, EMBEDDING[lang][0], max_features, embed_size,lang)
            fast_embedding_matrix = meta_embedding(tok, EMBEDDING[lang][1], max_features, embed_size,lang)


            # char_embedding = char_vectorizer.get_char_embedding(tok)

            # embedding_matrix = np.mean([glove_embedding_matrix, fast_embedding_matrix], axis=0)

            embedding_matrix = np.concatenate((glove_embedding_matrix, fast_embedding_matrix), axis=1)

            if type_model == 'three':
                # X_train_2 = train_new[train_new['label_quality'] == 'reliable']['small_title']
                X_train_3 = train_new[train_new['label_quality'] == 'reliable'] \
                    [['n_words', 'length', 'n_chars_word', 'n_capital_letters', 'n_numbers', 'small_length',
                      'small_n_chars_word', 'small_n_capital_letters', 'small_n_numbers',
                      'numbers', 'sum_numbers', 'mean_numbers']].values

                # X_train_2 = tok.texts_to_sequences(X_train_2)
                # X_train_2 = sequence.pad_sequences(X_train_2, maxlen=6)


                X_train, X_val, X_train_3, X_val_3, Y_train, Y_val = train_test_split(X_train, X_train_3, Y_train, train_size=0.9, random_state=233)

                train_generator = DataGenerator([X_train, X_train_3], Y_train, classes, batch_size=batch_size,mode=type_model)
                val_generator = DataGenerator([X_val, X_val_3], Y_val, classes, batch_size=batch_size,mode=type_model)

            else:

                X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.9, random_state=233)

                train_generator = DataGenerator(X_train, Y_train, classes, batch_size=batch_size)
                val_generator = DataGenerator(X_val, Y_val, classes, batch_size=batch_size)

            opt = RAdam(lr=1e-3)
            # opt = Nadam(lr=1e-3, schedule_decay=0.005)
            # opt = Adam(lr=1e-3)
            if type_model == 'small':
                model = get_small_model(maxlen, max_features, 2*embed_size, embedding_matrix, len(classes))

            elif type_model == 'three':
                model = get_three_entrys_model(maxlen, max_features, 2*embed_size, embedding_matrix, len(classes))

            else:
                model = get_model(maxlen, max_features, 2*embed_size, embedding_matrix, len(classes))
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

            print("Pré treinando")

            reduce_lr = ReduceLROnPlateau(
                            monitor  = 'val_loss',
                            factor   = 0.3,
                            patience = 1,
                            verbose  = 1,
                            mode     = 'auto',
                            epsilon  = 0.0001,
                            cooldown = 0,
                            min_lr   = 0
                        )
            early = EarlyStopping(monitor="val_loss", mode="min", patience=3)

            callbacks_list = [early, reduce_lr]

            model.fit_generator(generator=train_generator,
                                validation_data=val_generator,
                                callbacks=callbacks_list,
                                epochs=30,
                                use_multiprocessing=True,
                                workers=42)


            X_train = train_new['title']

            Y_train = train_new['category'].values

            tok, X_train = tokenize(X_train, X_test, max_features, maxlen, lang)
            glove_embedding_matrix = meta_embedding(tok, EMBEDDING[lang][0], max_features, embed_size,lang)
            fast_embedding_matrix = meta_embedding(tok, EMBEDDING[lang][1], max_features, embed_size,lang)

            # char_embedding = char_vectorizer.get_char_embedding(tok)

            # embedding_matrix = np.mean([glove_embedding_matrix, fast_embedding_matrix], axis=0)

            embedding_matrix = np.concatenate((glove_embedding_matrix, fast_embedding_matrix), axis=1)


            class_weights = class_weight.compute_class_weight('balanced',
                                                              classes,
                                                              Y_train)

            if type_model == 'three':
                # X_train_2 = train_new['small_title']
                X_train_3 = train_new[['n_words','length', 'n_chars_word','n_capital_letters','n_numbers','small_length',
                              'small_n_chars_word','small_n_capital_letters','small_n_numbers',
                              'numbers', 'sum_numbers', 'mean_numbers']].values

                # X_train_2 = tok.texts_to_sequences(X_train_2)
                # X_train_2 = sequence.pad_sequences(X_train_2, maxlen=6)

                X_test_small = test_new["small_title"]
                X_test_small = tok.texts_to_sequences(X_test_small)
                X_test_small = sequence.pad_sequences(X_test_small, maxlen=6)
                X_test_features =  test_new[['n_words','length', 'n_chars_word','n_capital_letters','n_numbers','small_length',
                              'small_n_chars_word','small_n_capital_letters','small_n_numbers',
                              'numbers', 'sum_numbers', 'mean_numbers']].values

                save_multi_inputs(X_test_small,X_test_features, lang)

                X_train, X_val, X_train_3, X_val_3, Y_train, Y_val = train_test_split(X_train, X_train_3, Y_train, train_size=0.9, random_state=233)

                train_generator = DataGenerator([X_train, X_train_3], Y_train, classes, batch_size=batch_size,mode=type_model)
                val_generator = DataGenerator([X_val, X_val_3], Y_val, classes, batch_size=batch_size,mode=type_model)
                model.get_layer('embedding_layer').set_weights([embedding_matrix])
                model.get_layer('small_embedding_layer').set_weights([embedding_matrix])

            else:

                X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.9, random_state=233, stratify=Y_train)

                train_generator = DataGenerator(X_train, Y_train, classes, batch_size=batch_size)
                val_generator = DataGenerator(X_val, Y_val, classes, batch_size=batch_size)

                model.layers[1].set_weights([embedding_matrix])


            opt = Adam(lr=0.001)

            model.compile(loss=label_smooth_loss, optimizer=opt, metrics=['accuracy'])

            filepath = '../models/' + lang + '_model_{epoch:02d}_{val_acc:.4f}.h5'
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max',
                                         save_weights_only=False)
            early = EarlyStopping(monitor="val_loss", mode="min", patience=3)

            # clr = CyclicLR(base_lr=0.0003, max_lr=0.001,
            #                step_size=35000, reduce_on_plateau=1, monitor='val_loss', reduce_factor=10)

            reduce_lr = ReduceLROnPlateau(
                            monitor  = 'val_loss',
                            factor   = 0.3,
                            patience = 1,
                            verbose  = 1,
                            mode     = 'auto',
                            epsilon  = 0.0001,
                            cooldown = 0,
                            min_lr   = 0
                        )

            callbacks_list = [checkpoint, early, reduce_lr]

            lookahead = Lookahead(k=5, alpha=0.5)  # Initialize Lookahead
            lookahead.inject(model)


            print("Treinando")

            model.fit_generator(generator=train_generator,
                                validation_data=val_generator,
                                callbacks=callbacks_list,
                                class_weight=class_weights,
                                epochs=50,
                                use_multiprocessing=True,
                                workers=42)


        else:
            class_weights = class_weight.compute_class_weight('balanced',
                                                              classes,
                                                              Y_train)

            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.9, random_state=233)

            train_generator = DataGenerator(X_train, Y_train, classes, batch_size=batch_size)
            val_generator = DataGenerator(X_val, Y_val, classes, batch_size=batch_size)

            # opt = RAdam(lr=1e-3)
            # opt = Nadam(lr=1e-3)
            opt = Adam(lr=1e-3)
            if type_model == 'colab':
                model = get_small_model(maxlen, max_features, embed_size, embedding_matrix, len(classes))
            elif type_model == 'three':
                model = get_three_entrys_model(maxlen, max_features, embed_size, embedding_matrix, len(classes))

            else:
                model = get_model(maxlen,max_features,embed_size,embedding_matrix,len(classes))
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

            lookahead = Lookahead(k=5, alpha=0.5)  # Initialize Lookahead
            lookahead.inject(model)

            filepath = '../models/' + lang + '_model_{epoch:02d}_{val_acc:.4f}.h5'
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max',
                                         save_weights_only=False)
            early = EarlyStopping(monitor="val_loss", mode="min", patience=3)

            clr = CyclicLR(base_lr=0.000001, max_lr=0.001,
                           step_size=35000.)

            # reduce_lr = ReduceLROnPlateau(
            #                 monitor  = 'val_loss',
            #                 factor   = 0.3,
            #                 patience = 1,
            #                 verbose  = 1,
            #                 mode     = 'auto',
            #                 epsilon  = 0.0001,
            #                 cooldown = 0,
            #                 min_lr   = 0
            #             )

            callbacks_list = [checkpoint, early,clr]

            print("Treinando")

            model.fit_generator(generator=train_generator,
                                validation_data=val_generator,
                                callbacks=callbacks_list,
                                class_weight=class_weights,
                                epochs=50,
                                use_multiprocessing=True,
                                workers=42)

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Predict script')


    parser.add_argument('--model', help='Local of training', default='normal')
    parser.add_argument('--pre', help='Pretraining with only reliable values', default=False, type=bool)
    parser.add_argument('--language', help='Training only in a specific language', default='both')


    return parser.parse_args(args)

if __name__ == '__main__':
    train = pd.read_csv("../../dados/train.csv")
    test = pd.read_csv("../../dados/test.csv")

    args = sys.argv[1:]
    args = parse_args(args)

    EMBEDDING = {"spanish": ["../../../harold/word_embeddings/espanhol/glove-sbwc.i25.vec",
                             "../../../harold/word_embeddings/espanhol/fasttext-nlarge-suc.vec"],

                 "portuguese": ["../../../harold/word_embeddings/portugues/glove_s300.txt",
                                "../../../harold/word_embeddings/portugues/skip_s300.txt"]}

    if args.language == 'both':
        languages = ['portuguese', 'spanish']
    else:
        languages = []
        languages.append(args.language)



    training(languages,EMBEDDING,train,test,args.model, args.pre)
