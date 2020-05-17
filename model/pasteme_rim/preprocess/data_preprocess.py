"""
数据预处理
"""

import jieba
import re
import typing
import pandas as pd
import numpy as np
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


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
