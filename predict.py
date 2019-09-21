import os

import pandas as pd
import argparse
import sys

from utils.layers import  AttentionWithContext
from utils.utils import focal_loss, label_smooth_loss
import pickle
import keras
from keras_radam import RAdam
import numpy as np


def predict(languages, pt_weight,es_weight,train,test,name,type_model):

    submission = pd.DataFrame()

    for lang in languages:
        train_new = train[train["language"] == lang]
        test_new = test[test["language"] == lang]

        classes = train_new["category"].unique()


        with open('../tokenizers/' + lang +'_tokenizer.pickle', 'rb') as handle:
            test_tokenized = pickle.load(handle)

        if type_model == 'three':
            with open('../tokenizers/' + lang + '_small_tokenizer.pickle', 'rb') as handle:
                test_tokenized_small = pickle.load(handle)

            with open('../tokenizers/' + lang + '_features_tokenizer.pickle', 'rb') as handle:
                test_tokenized_features = pickle.load(handle)


        custom_objects = {
            'RAdam': RAdam,
            'focal_loss': focal_loss,
            'AttentionWithContext' : AttentionWithContext,
            'label_smooth_loss' : label_smooth_loss
        }

        if lang == 'portuguese':
            model_path = pt_weight
            val_acc = name.split('_')[1]
        else:
            model_path = es_weight
            val_acc = name.split('_')[2]


        model = keras.models.load_model(model_path, custom_objects=custom_objects)

        if type_model == 'three':
            model_pred = model.predict([test_tokenized,test_tokenized_features], batch_size=4096)
        else:
            model_pred = model.predict(test_tokenized, batch_size=4096)

        with open('predictions/' + lang + '_' + val_acc + '_.pickle', 'wb') as handle:
            pickle.dump(model_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)


        model_pred = model_pred.argmax(axis=1)

        model_pred_class = [0] * len(model_pred)

        for i, value in enumerate(model_pred):
            model_pred_class[i] = classes[value]

        test_new['category'] = model_pred_class
        submission = submission.append(test_new[['id','category']])


    submission.to_csv('submissions/' + name + '.csv', index=False)

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Predict script')


    parser.add_argument('--pt_weight', help='Path to portuguese weight')
    parser.add_argument('--es_weight', help='Path to spanish weight')
    parser.add_argument('--name', help="Name of csv submission", default="submission")
    parser.add_argument("--cpu", default=False, type=bool)
    parser.add_argument("--model", default='normal', help='Type of model')

    return parser.parse_args(args)


if __name__ == '__main__':
    train = pd.read_csv("../../dados/train.csv")
    test = pd.read_csv("../../dados/test.csv")

    EMBEDDING = {"spanish": "../../../harold/word_embeddings/espanhol/glove-sbwc.i25.vec",
                 "portuguese": "../../../harold/word_embeddings/espanhol/glove-sbwc.i25.vec"}

    languages = ['portuguese', 'spanish']

    args = sys.argv[1:]
    args = parse_args(args)

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    predict(languages,args.pt_weight,args.es_weight,train,test,args.name,args.model)
