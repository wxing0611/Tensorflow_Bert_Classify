# coding=utf-8
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import six
import unicodedata
import collections
import tensorflow as tf


def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
    """
        这里主要核对 如果要求全是小写字母，那么提前准备小写模型及参数， 反之，准备大写模型及参数
        如果训练的数据集是中文，这里则无意义
    :param do_lower_case: 是否小写
    :param init_checkpoint: 已有训练好的模型存储路径
    :return:
    """
    if not init_checkpoint:
        return

    m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
    if m is None:
        return

    model_name = m.group(1)  # 获取上面末尾文件夹名 uncased_L-12_H-768_A-12
    lower_models = [
        "uncased_L-24_H-1024_A-16",
        "uncased_L-12_H-768_A-12",
        "multilingual_L-12_H-768_A-12",
        "chinese_L-12_H-768_A-12"
    ]
    cased_models = [
        "cased_L-12_H-768_A-12",
        "cased_L-24_H-1024_A-16",
        "multi_cased_L-12_H-768_A-12"
    ]
    is_bad_config = False
    if model_name in lower_models and not do_lower_case:
        is_bad_config = True
        actual_flag = "False"
        case_name = "lowercased"
        opposite_flag = "True"

    if model_name in cased_models and do_lower_case:
        is_bad_config = True
        actual_flag = "True"
        case_name = "cased"
        opposite_flag = "False"

    if is_bad_config:
        raise ValueError(
            "You passed in `--do_lower_case=%s` with `--init_checkpoint=%s`. "
            "However, `%s` seems to be a %s model, so you "
            "should pass in `--do_lower_case=%s` so that the fine-tuning matches "
            "how the model was pre-training. If this error is wrong, please "
            "just comment out this check." % (actual_flag, init_checkpoint,
                                              model_name, case_name, opposite_flag))


def convert_to_unicode(text):
    """ 如果text是二进制，则decode('utf-8') """
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """ 如果text是二进制，则decode('utf-8'), """
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        # elif isinstance(text, unicode):
        #     return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """加载文章词集文件，并统计保存为字典 {词：索引}"""
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def convert_by_vocab(vocab, items):
    """词转索引 或 索引转词"""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
    """空格切分文本"""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class FullTokenizer(object):
    """集成分词器"""
    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)  # 加载词文件，并统计返回词字典 {词：索引}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}  # {索引：词}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)  # 定义基础分词器
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)  # 定义Wordpiece英文细分词器

    def tokenize(self, text):
        split_tokens = []
        # 对输入text进行基础分词
        for token in self.basic_tokenizer.tokenize(text):
            # 对英文词进行二次分词，如："unaffable" ==> ["un", "##aff", "##able"]
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        # 给定词，返回对应词索引
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        # 给定词索引，返回对应词
        return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
    """基础分词类：中文单字，英文单词，标点符号分割"""
    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.
        Args:
          do_lower_case: 是否是小写词输入
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """句子text进行分词，返回分词后的列表"""
        text = convert_to_unicode(text)  # 转换成unicode字符
        text = self._clean_text(text)  # 将 \r\n\r 统一换成 空格
        # 如果是中文字符则两边添加空格，其他字符则不需要
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)  # 用空格切分text，结果为列表
        split_tokens = []
        for token in orig_tokens:
            # 遍历每个词
            if self.do_lower_case:
                # 是否小写
                token = token.lower()
                token = self._run_strip_accents(token)  # 去除 Mn 类的字符
            split_tokens.extend(self._run_split_on_punc(token))  # 标点符号再次切分
        output_tokens = whitespace_tokenize(" ".join(split_tokens))  # 用空格切分text，结果为列表
        return output_tokens

    def _run_strip_accents(self, text):
        """从一段文字中去除 Mn 类型的字符"""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """以标点符号来分割"""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                # 是标点符号
                output.append([char])
                start_new_word = True
            else:
                # 不是标点符号
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """如果是中文字符则两边添加空格，其他字符则不需要"""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """是否是正文字符"""
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """对文本执行无效字符删除和并将 \r\n\r 统一换成空格 """
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """ WordPiece tokenziation. 将一个英文长词分词多个英文短词组合 """

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab  # 词字典 {词：索引}
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """ 将一个长词分词多个短词组合  "unaffable" ==> ["un", "##aff", "##able"] """
        text = convert_to_unicode(text)
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """检查是否是 空白字符"""
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """检查是否是 控制字符"""
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):  # ctrl+c 或 ctrl+f
        return True
    return False


def _is_punctuation(char):
    """是否是标点符号，是为True，不是为False"""
    cp = ord(char)
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
