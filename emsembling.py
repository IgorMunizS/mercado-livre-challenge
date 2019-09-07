import pickle
import os
import pandas as pd
import sys
import argparse

def emsemble(folder):

    predictions = os.listdir(folder)
    train = pd.read_csv("../../dados/train.csv")
    test = pd.read_csv("../../dados/test.csv")

    submission = pd.DataFrame()
    name = 'emsembling_'

    for language in ['portuguese','spanish']:
        model_predict_list = []
        val_acc_list = []

        train_new = train[train["language"] == language]
        test_new = test[test["language"] == language]

        classes = train_new["category"].unique()

        for predict in predictions:
            if predict.split('.')[1] == 'pickle':
                lang = predict.split('_')[0]
                val_acc = predict.split('_')[1]

                if lang == language:
                    with open('predictions/' + predict, 'rb') as handle:
                        model_predict = pickle.load(handle)
                    model_predict_list.append(model_predict*val_acc)
                    val_acc_list.append(val_acc)


                    final_predict= sum(model_predict_list) / sum(val_acc_list)

                    model_pred = final_predict.argmax(axis=1)

                    model_pred_class = [0] * len(final_predict)

                    for i, value in enumerate(model_pred):
                        model_pred_class[i] = classes[value]

                    test_new['category'] = model_pred_class
                    submission = submission.append(test_new[['id', 'category']])

                    name = name + lang + '_'.join(val_acc_list)

    submission.to_csv('submissions/' + name + '.csv', index=False)

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Emsemble results script')


    parser.add_argument('--folder', help='Path to pickle predictions', default='predictions/')


    return parser.parse_args(args)

if __name__ == '__main__':

    args = sys.argv[1:]
    args = parse_args(args)

    emsemble(args.folder)


