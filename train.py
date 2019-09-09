import pandas as pd
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import Adam, Nadam
from sklearn.model_selection import train_test_split

from generator import DataGenerator
from model import get_model, get_small_model, get_attention_model
from utils import tokenize, embedding, focal_loss, CyclicLR
from sklearn.utils import class_weight
from keras_radam import RAdam
import argparse
import sys
import numpy as np


def training(languages, EMBEDDING,train,test,env):

    for lang in languages:
        train_new = train[train["language"] == lang]
        test_new = test[test["language"] == lang]

        # X_train = train_new['title'].str.lower()
        X_train = train_new['title']

        Y_train = train_new['category'].values
        classes = train_new["category"].unique()

        X_test = test_new["title"]

        # X_test = test_new["title"].str.lower()

        class_weights = class_weight.compute_class_weight('balanced',
                                                          classes,
                                                          Y_train)

        max_features = 100000
        maxlen = 30
        embed_size = 300

        tok, X_train = tokenize(X_train,X_test,max_features,maxlen,lang)
        embedding_matrix = embedding(tok,EMBEDDING[lang][0],max_features,embed_size)
        embedding_matrix_1 = embedding(tok,EMBEDDING[lang][1],max_features,embed_size)

        embedding_matrix = np.mean([embedding_matrix, embedding_matrix_1], axis=0)

        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.9, random_state=233)

        train_generator = DataGenerator(X_train, Y_train, classes, batch_size=4096)
        val_generator = DataGenerator(X_val, Y_val, classes, batch_size=4096)

        # opt = RAdam(lr=1e-3, min_lr=1e-5)
        opt = Nadam(lr=1e-5)
        if env == 'colab':
            model = get_small_model(maxlen, max_features, embed_size, embedding_matrix, len(classes))
        else:
            model = get_model(maxlen,max_features,embed_size,embedding_matrix,len(classes))
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        filepath = '../models/' + lang + '_model_{epoch:02d}_{val_acc:.4f}.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max',
                                     save_weights_only=False)
        early = EarlyStopping(monitor="val_loss", mode="min", patience=3)

        clr = CyclicLR(base_lr=0.00001, max_lr=0.001,
                       step_size=20000.)

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


    parser.add_argument('--env', help='Local of training', default='v100')


    return parser.parse_args(args)

if __name__ == '__main__':
    train = pd.read_csv("../../dados/train.csv")
    test = pd.read_csv("../../dados/test.csv")

    EMBEDDING = {"spanish": ["../../../harold/word_embeddings/espanhol/glove-sbwc.i25.vec",
                             "../../../harold/word_embeddings/espanhol/fasttext-sbwc.vec"],

                 "portuguese": ["../../../harold/word_embeddings/portugues/glove_s300.txt",
                                "../../../harold/word_embeddings/portugues/skip_s300.txt"]}

    languages = ['portuguese', 'spanish']

    args = sys.argv[1:]
    args = parse_args(args)

    training(languages,EMBEDDING,train,test,args.env)
