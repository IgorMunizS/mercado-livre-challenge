import os

import pandas as pd
import argparse
import sys

from utils import tokenize, embedding, focal_loss
import pickle
import keras

def predict(languages, pt_weight,es_weight,train,test,name):

    submission = pd.DataFrame()

    for lang in languages:
        train_new = train[train["language"] == lang]
        test_new = test[test["language"] == lang]

        classes = train_new["category"].unique()


        with open('../tokenizers/' + lang +'_tokenizer.pickle', 'rb') as handle:
            test_tokenized = pickle.load(handle)


        custom_objects = {
            'focal_loss': focal_loss,
        }

        if lang == 'portuguese':
            model_path = pt_weight
        else:
            model_path = es_weight

        model = keras.models.load_model(model_path, custom_objects=custom_objects)

        model_pred = model.predict(test_tokenized, batch_size=4096)
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

    predict(languages,args.pt_weight,args.es_weight,train,test,args.name)
