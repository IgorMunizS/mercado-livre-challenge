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
    df['first_word'] = df['title'].apply(lambda x: ' '.join(x.split(' ')[0]))


    df['n_words'] = df['title'].apply(lambda x: len(x.split(' ')))
    df['length'] = df['title'].apply(lambda x: len(x))
    df['n_chars_word'] = df['length'] / df['n_words']
    df['n_capital_letters'] = df['title'].apply(lambda x: len(re.findall(r'[A-Z]', x)))
    df['n_numbers'] = df['title'].apply(lambda x: len(re.findall(r'[0-9]', x)))

    df['n_0'] = df['title'].apply(lambda x: len(re.findall(r'0', x)))
    df['n_1'] = df['title'].apply(lambda x: len(re.findall(r'1', x)))
    df['n_2'] = df['title'].apply(lambda x: len(re.findall(r'2', x)))
    df['n_3'] = df['title'].apply(lambda x: len(re.findall(r'3', x)))
    df['n_4'] = df['title'].apply(lambda x: len(re.findall(r'4', x)))
    df['n_5'] = df['title'].apply(lambda x: len(re.findall(r'5', x)))
    df['n_6'] = df['title'].apply(lambda x: len(re.findall(r'6', x)))
    df['n_7'] = df['title'].apply(lambda x: len(re.findall(r'7', x)))
    df['n_8'] = df['title'].apply(lambda x: len(re.findall(r'8', x)))
    df['n_9'] = df['title'].apply(lambda x: len(re.findall(r'9', x)))

    df['small_length'] = df['small_title'].apply(lambda x: len(x))
    df['small_n_chars_word'] = df['small_length'] / 5
    df['small_n_capital_letters'] = df['small_title'].apply(lambda x: len(re.findall(r'[A-Z]', x)))
    df['small_n_numbers'] = df['small_title'].apply(lambda x: len(re.findall(r'[0-9]', x)))

    df['first_word_length'] = df['first_word'].apply(lambda x: len(x))

    # df['small_n_capital_letters'] = df['small_title'].apply(lambda x: len(re.findall(r'[A-Z]', x)))
    # df['small_n_numbers'] = df['small_title'].apply(lambda x: len(re.findall(r'[0-9]', x)))

    # df['numbers'] = df['title'].apply(lambda x: get_all_numbers(x))
    # df['sum_numbers'] = df['title'].apply(lambda x: sum_all_numbers(x))
    # df['mean_numbers'] = df['title'].apply(lambda x: mean_all_numbers(x))


    return df