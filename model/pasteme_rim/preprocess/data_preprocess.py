"""
数据预处理
"""

import jieba
import re
import typing
import pandas as pd
from zhon import hanzi


def extract_chinese(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    只保留中文
    :param raw_df: columns = ['text', 'label']
    :return: df with chinese only
    """

    df = raw_df.copy()

    zh_pattern = re.compile('[{}]'.format(hanzi.characters))
    df['text'] = df['text'].apply(lambda x: str.join('', zh_pattern.findall(x)))

    return df


def tokenize(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    使用 jieba 分词
    :param raw_df: source DataFrame
    :return: df with column named "tokens"
    """

    df = raw_df.copy()
    df['tokens'] = df['text'].apply(
        lambda x: list(jieba.cut(x))
    )

    return df


def tokens_to_ids(raw_df: pd.DataFrame, word2idx: typing.Dict[str, int]) -> pd.DataFrame:
    df = raw_df.copy()
    df['text'] = df['tokens'].apply(lambda x: [word2idx.get(each, 0) for each in x])
    return df


def balanced_sampling(raw_df: pd.DataFrame, down_sampling: bool = False) -> pd.DataFrame:
    """
    上采样或下采样，默认上采样
    :param raw_df: df with column named 'label'
    :param down_sampling: using down sampling
    :return: balanced df by up sampling
    """

    df = raw_df.copy()

    labels = df['label'].unique()

    label_to_data = {}
    for label in labels:
        label_to_data[label] = df[df['label'] == label]

    standard_size = len(label_to_data[labels[0]])

    for label in labels:
        length = len(label_to_data[label])
        if down_sampling ^ (length > standard_size):
            standard_size = length

    result_df_list = []

    for label in labels:
        length = len(label_to_data[label])
        if length > standard_size:  # 因为是上采样，所以这个分支永远不会被走到，保险起见还是写上
            result_df_list.append(label_to_data[label].sample(n=standard_size))
        else:
            result_df_list.append(label_to_data[label])

            if length < standard_size:
                result_df_list.append(label_to_data[label].sample(n=standard_size - length, replace=True))

    return pd.concat(result_df_list, axis=0).sample(frac=1.)
