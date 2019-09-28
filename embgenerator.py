import argparse
import sys
import pandas as pd
from utils.preprocess import clean_numbers, clean_text, replace_typical_misspell, normalize_title, RemoveStopWords
from tqdm import tqdm
tqdm.pandas()
from gensim.models.fasttext import FastText as FT_gensim
from gensim.test.utils import datapath

def get_sentences(train,test,lang):
    train_new = train[train["language"] == lang]
    test_new = test[test["language"] == lang]

    train_new['title'] = train_new['title'].str.lower()
    test_new['title'] = test_new['title'].str.lower()

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

    return (train_new['title'].tolist() + test_new["title"].tolist())


def generate_corpus(sentences,name):

    with open('embedding/corpus_' + name, 'w') as f:
        for sentence in tqdm(sentences):
            f.write(sentence + '\n')

def generate_model(lang):

    corpus_file = datapath('embedding/corpus_' + 'lang')

    model_gensim = FT_gensim(size=300)

    # build the vocabulary
    model_gensim.build_vocab(corpus_file=corpus_file)

    # train the model
    model_gensim.train(
        corpus_file=corpus_file, epochs=model_gensim.epochs,
        total_examples=model_gensim.corpus_count, total_words=model_gensim.corpus_total_words
    )

    model_gensim.save('embedding/fasttext_' + lang + '.vec')

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Predict script')

    parser.add_argument('--task', help='Type of task to execute', default='corpus')
    parser.add_argument('--language', help='Training only in a specific language', default='both')
    parser.add_argument('--data_folder', default='../../dados/')

    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)

    train = pd.read_csv(args.data_folder + "train.csv")
    test = pd.read_csv(args.data_folder + "test.csv")

    if args.language == 'both':
        languages = ['portuguese', 'spanish']
    else:
        languages = []
        languages.append(args.language)

    if args.task == 'corpus':
        for lang in languages:
            sentences = get_sentences(train,test,lang)
            generate_corpus(sentences,lang)

    if args.task == 'model':
        for lang in languages:
            generate_model(lang)