from pre_models import get_bert_model

from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight
import argparse
import sys
from preprocess import clean_numbers, clean_text, replace_typical_misspell
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

def finetunning(languages, model,train,test,env):

    for lang in languages:
        train_new = train[train["language"] == lang]
        test_new = test[test["language"] == lang]

        train_new['title'] = train_new['title'].str.lower()
        test_new['title'] = test_new['title'].str.lower()

        train_new["title"] = train_new["title"].progress_apply(lambda x: clean_numbers(x))
        train_new["title"] = train_new["title"].progress_apply(lambda x: replace_typical_misspell(x, lang))
        train_new["title"] = train_new["title"].progress_apply(lambda x: clean_text(x))

        test_new["title"] = test_new["title"].progress_apply(lambda x: clean_numbers(x))
        test_new["title"] = test_new["title"].progress_apply(lambda x: replace_typical_misspell(x, lang))
        test_new["title"] = test_new["title"].progress_apply(lambda x: clean_text(x))

        X_train = train_new['title']

        Y_train = train_new['category'].values
        classes = train_new["category"].unique()

        X_test = test_new["title"]



        class_weights = class_weight.compute_class_weight('balanced',
                                                          classes,
                                                          Y_train)

        max_features = 100000
        maxlen = 30
        embed_size = 300
        batch_size = 512


        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.9, random_state=233)

        if model == 'bert':
            model = get_bert_model(batch_size,maxlen,len(X_train),(X_val,Y_val),'../models/')

        model.fit(X_train,Y_train)


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Predict script')


    parser.add_argument('--env', help='Local of training', default='v100')
    parser.add_argument('--model', help='Pretrained model to use', default='bert')

    return parser.parse_args(args)

if __name__ == '__main__':
    train = pd.read_csv("../../dados/train.csv")
    test = pd.read_csv("../../dados/test.csv")

    languages = ['portuguese', 'spanish']

    args = sys.argv[1:]
    args = parse_args(args)

    finetunning(languages,args.model,train,test,args.env)
