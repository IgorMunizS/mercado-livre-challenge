import pandas as pd
import re

def get_all_numbers(title):
    numbers_list = re.findall('\d+', title)
    if len(numbers_list) > 0:
        return int(''.join(numbers_list))
    else:
        return 0

def sum_all_numbers(title):
    numbers_list = re.findall('\d+', title)
    if len(numbers_list) > 0:
        numbers_list = [int(x) for x in numbers_list]
        return sum(numbers_list)
    else:
        return 0

def mean_all_numbers(title):
    numbers_list = re.findall('\d+', title)
    if len(numbers_list) > 0:
        numbers_list = [int(x) for x in numbers_list]
        return sum(numbers_list) / len(numbers_list)
    else:
        return 0



def build_features(df):
    print("Gerando Features")
    df['small_title'] = df['title'].apply(lambda x: ' '.join(x.split(' ')[:5]))
    df['n_words'] = df['title'].apply(lambda x: len(x.split(' ')))
    df['length'] = df['title'].apply(lambda x: len(x))
    # df['n_chars_word'] = df['length'] / df['n_words']
    df['n_capital_letters'] = df['title'].apply(lambda x: len(re.findall(r'[A-Z]', x)))
    df['n_numbers'] = df['title'].apply(lambda x: len(re.findall(r'[0-9]', x)))
    df['small_length'] = df['small_title'].apply(lambda x: len(x))
    # df['small_n_chars_word'] = df['small_length'] / 5
    df['small_n_capital_letters'] = df['small_title'].apply(lambda x: len(re.findall(r'[A-Z]', x)))
    df['small_n_numbers'] = df['small_title'].apply(lambda x: len(re.findall(r'[0-9]', x)))

    # df['numbers'] = df['title'].apply(lambda x: get_all_numbers(x))
    # df['sum_numbers'] = df['title'].apply(lambda x: sum_all_numbers(x))
    # df['mean_numbers'] = df['title'].apply(lambda x: mean_all_numbers(x))


    return df