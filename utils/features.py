import pandas as pd
import re


def build_features(df):
    df['small_title'] = df['title'].apply(lambda x: ' '.join(x.split(' ')[:5]))
    df['n_words'] = df['title'].apply(lambda x: len(x.split(' ')))
    df['length'] = df['title'].apply(lambda x: len(x))
    df['n_capital_letters'] = df['title'].apply(lambda x: len(re.findall(r'[A-Z]', x)))
    df['n_numbers'] = df['title'].apply(lambda x: len(re.findall(r'[0-9]', x)))
    df['small_length'] = df['small_title'].apply(lambda x: len(x))
    df['small_n_capital_letters'] = df['small_title'].apply(lambda x: len(re.findall(r'[A-Z]', x)))
    df['small_n_numbers'] = df['small_title'].apply(lambda x: len(re.findall(r'[0-9]', x)))

    return df