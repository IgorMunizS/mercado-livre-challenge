import pandas as pd
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from generator import DataGenerator
from model import get_model
from utils import tokenize, embedding, focal_loss
from sklearn.utils import class_weight


def training(languages, EMBEDDING,train,test):

    for lang in languages:
        train_new = train[train["language"] == lang]
        test_new = test[test["language"] == lang]

        X_train = train_new['title'].str.lower()
        Y_train = train_new['category'].values
        classes = train_new["category"].unique()


        X_test = test_new["title"].str.lower()

        class_weights = class_weight.compute_class_weight('balanced',
                                                          classes,
                                                          Y_train)

        max_features = 100000
        maxlen = 30
        embed_size = 300

        tok, X_train = tokenize(X_train,X_test,max_features,maxlen,lang)
        embedding_matrix = embedding(tok,EMBEDDING[lang],max_features,embed_size)

        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.9, random_state=233)

        train_generator = DataGenerator(X_train, Y_train, classes, batch_size=4096)
        val_generator = DataGenerator(X_val, Y_val, classes, batch_size=4096)


        model = get_model(maxlen,max_features,embed_size,embedding_matrix,len(classes))
        model.compile(loss=[focal_loss], optimizer=Adam(lr=1e-3), metrics=['accuracy'])

        filepath = '../models/' + lang + '_model_{epoch:02d}_{val_acc:.4f}.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max',
                                     save_weights_only=False)
        early = EarlyStopping(monitor="val_loss", mode="min", patience=2)

        reduce_lr = ReduceLROnPlateau(
                        monitor  = 'val_loss',
                        factor   = 0.1,
                        patience = 1,
                        verbose  = 1,
                        mode     = 'auto',
                        epsilon  = 0.0001,
                        cooldown = 0,
                        min_lr   = 0
                    )

        callbacks_list = [checkpoint, early,reduce_lr]

        print("Treinando")

        model.fit_generator(generator=train_generator,
                            validation_data=val_generator,
                            callbacks=callbacks_list,
                            class_weight=class_weights,
                            epochs=50,
                            use_multiprocessing=True,
                            workers=42)

if __name__ == '__main__':
    train = pd.read_csv("../../dados/train.csv")
    test = pd.read_csv("../../dados/test.csv")

    EMBEDDING = {"spanish": "../../../harold/word_embeddings/espanhol/glove-sbwc.i25.vec",
                 "portuguese": "../../../harold/word_embeddings/espanhol/glove-sbwc.i25.vec"}

    languages = ['portuguese', 'spanish']

    training(languages,EMBEDDING,train,test)
