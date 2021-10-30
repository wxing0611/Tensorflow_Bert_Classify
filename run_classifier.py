# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

""" 启动必要要求参数： (参数名, 默认值，备注) """
# 数据存放位置的文件夹
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")
# bert模型需要的配置参数
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")
# 任务名称
flags.DEFINE_string("task_name", None, "The name of the task to train.")
# 所有出现的词的汇总文件
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
# 结果输出文件夹
flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

""" 其他参数非必要传参数，使用默认值也可 """
# 预训练好的模型，加载模型参数作为初始化参数
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")
# 是否全为小写，True全转换为小写，False不做转换
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
# 一句话输入最大词数
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")
# 是否训练
flags.DEFINE_bool("do_train", False, "Whether to run training.")
# 是否评估
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
# 是否预测
flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")
# 训练集batch
flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
# 评估集batch
flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")
# 预测集batch
flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")
# 学习率
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
# 训练轮次
flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")
# warmup目的是让前面的batch的学习率降低，后面的batch的学习率恢复之前设置
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")
# 多久保存一次模型
flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")
# 训练多少次后做一次评估
flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

""" 下面TPU参数设置 """
# 是否使用TPU
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """ 单次输入的数据与标签 """

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """ 单次输入的数据特征集合 """

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """读取tsv文件数据"""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """根据文件存储形式进行修改"""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


## 不同的数据集，需要在这里对源数据进行处理
class NewsProcessor(DataProcessor):
    """ News新闻数据提取&封装 """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # line = [标签，第一个句子的id，第二个句子的id，第一个句子文本，第二个句子文本]
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])  # 转成中文字符
            # text_b = tokenization.convert_to_unicode(line[2])  # 转成中文字符
            # 获取标签
            if set_type == "test":
                label = "0"
            else:
                label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    """
        输入特征封装：单个样本进行转换
    :param ex_index: 索引：第几个句子
    :param example: 数据：句子
    :param label_list: 类别集
    :param max_seq_length: 最大长度
    :param tokenizer: 分词对象
    :return:
    """
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):  # 构建标签
        label_map[label] = i  # {标签：索引}

    tokens_a = tokenizer.tokenize(example.text_a)  # 对第一句话分词
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)  # 如果有第二句，则对第二句话分词

    if tokens_b:
        # 两句话输入结构： [CLS] [句1词1] [句1词2] [句1词3] [SEP] [句2词1] [句2词2] [句2词3] [句2词4] [SEP]
        # Account for [CLS], [SEP], [SEP] with "- 3" #保留3个特殊字符
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)  # 如果这俩太长了就截断操作，没有则不用做任何处理
    else:
        # 一句话输入结构：[CLS] [句1词1] [句1词2] [句1词3] [SEP]
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]  # 如果样本太长了就截断操作，没有则不用做任何处理

    # 转换成Bert需要的格式:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1 #表示来自哪句话
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0

    tokens = []  # 记录 字符
    segment_ids = []  # 记录 每个字符来源于第几个句子
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)  # 将对应字转换索引ID

    # 当输入长度未达标，则用0填充。并用input_mask记录1是真实的词，0是填充值。
    input_mask = [1] * len(input_ids)  # 由于后续可能会有补齐操作，设置了一个mask目的是让attention只能放到mask为1的位置
    while len(input_ids) < max_seq_length:  # PAD的长度取决于设置的最大长度
        input_ids.append(0)  # 词列表
        input_mask.append(0)  # 填充词标记列表
        segment_ids.append(0)  # 句子索引列表

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]  # 将标签集中的标签页转成索引
    if ex_index < 5:  # 打印一些例子
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    # 输入特征封装
    feature = InputFeatures(
        input_ids=input_ids,  # 词列表
        input_mask=input_mask,  # 填充词标记列表
        segment_ids=segment_ids,  # 句子索引列表
        label_id=label_id,  # 对应标签索引
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """
        将所有数据转换成TFRecord file，模型训练读取速度更快
    :param examples: 训练数据集
    :param label_list: 类别集
    :param max_seq_length: 最大长度
    :param tokenizer: 词与id
    :param output_file: 输出文件
    :return:
    """
    # tf模块读取数据更快
    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        # 遍历处理所有数据
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        # 一个样本数据转换处理
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)

        # 数据类型转换
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)  # 词列表
        features["input_mask"] = create_int_feature(feature.input_mask)  # 填充词标记列表
        features["segment_ids"] = create_int_feature(feature.segment_ids)  # 句子索引列表
        features["label_ids"] = create_int_feature([feature.label_id])  # 对应标签索引
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])
        # 数据序列化写入tf
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """
        Creates an `input_fn` closure to be passed to TPUEstimator.
    :param input_file: 训练数据集文件
    :param seq_length: 最大长度
    :param is_training: 是否训练
    :param drop_remainder:
    :return:
    """
    # 初始化
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
        截断输出词序列
    :param tokens_a: bert句子1
    :param tokens_b: bert句子2
    :param max_length: 输入序列最大长度
    :return:
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """
        创建一个bert+分类模型
    :param bert_config: bert配置文件信息
    :param is_training: 是否训练，True/False
    :param input_ids:词列表
    :param input_mask:填充词标记列表
    :param segment_ids:句子索引列表
    :param labels:对应标签索引
    :param num_labels:标签类别数量
    :param use_one_hot_embeddings:是否用tpu
    :return: batch输出的平均损失，batch内所有输入的损失，计算结果，结果softmax转换成概率
    """
    """1详细构建Bert模型"""
    model = modeling.BertModel(
        config=bert_config,  # bert配置文件信息
        is_training=is_training,  # 是否训练，True/False
        input_ids=input_ids,  # 词列表（8,128）
        input_mask=input_mask,  # 填充词标记列表（8,128）
        token_type_ids=segment_ids,  # 句子索引列表（8,128）
        use_one_hot_embeddings=use_one_hot_embeddings)  # tpu才用use_one_hot_embeddings

    output_layer = model.get_pooled_output()  # 返回最后一层输出 [batch, embedding_size]

    """2结合bert模型输出，接上自建(无激活函数)的全连接层+softmax，做最后的分类"""
    # 输出层类别数=隐藏层数量
    hidden_size = output_layer.shape[-1].value
    # 全连接层w和b的参数
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        # 下面相当于做了一个无激活函数的全连接，输出维度为num_labels
        # 损失值计算
        if is_training:
            # 0.1 忽略
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        # [batch, hidden_size] * [hidden_size, num_labels] = [batch, num_labels]
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)  # transpose_b=True, b在乘法之前转置.
        logits = tf.nn.bias_add(logits, output_bias)  # +b

        # 类别概率
        probabilities = tf.nn.softmax(logits, axis=-1)  # 每个类别概率值
        # 计算损失值
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)  # batch内所有输入的损失
        loss = tf.reduce_mean(per_example_loss)  # batch平均损失 对batch所有输入得到的输出的损失求均

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """
        Bert模型构建 & 加载模型参数
    :param bert_config: bert配置文件信息
    :param num_labels: 标签类别数量
    :param init_checkpoint: 别人训练好的模型参数位置
    :param learning_rate: 学习率
    :param num_train_steps: 训练次数
    :param num_warmup_steps: 前几个batch降低学习率
    :param use_tpu: 是否用tpu
    :param use_one_hot_embeddings: 是否用tpu
    :return:output_spec
    """
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """构建模型，加载参数，设定优化器"""
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]  # 词列表
        input_mask = features["input_mask"]  # 填充词标记列表
        segment_ids = features["segment_ids"]  # 句子索引列表
        label_ids = features["label_ids"]  # 对应标签索引
        # is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # 核心：Bert模型构建 =（batch输出的平均损失，batch内所有输入的损失，计算结果，结果softmax转换成概率）
        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids,
            label_ids, num_labels, use_one_hot_embeddings)

        # 加载已训练好的模型参数
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        # 加载已训练好的模型参数
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            # 训练，创建优化器
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)  # 设定INFO日志级别
    # 这里对不同的数据集，数据位置，数据结构，采用不同的processor
    processors = {
        # 任务名：数据处理函数
        "news": NewsProcessor,
    }
    # 针对大小写校验核对： 如果要求全是小写字母，那么提前准备小写模型及参数， 反之，准备大写模型及参数。如果错位，则会报错。
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.init_checkpoint)
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")
    # 加载模型配置文件参数
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    # 配置文件参数是别人已经训练好的预训练模型对应的参数，设定Bert最大支持一句话512词，如果设置参数大于512则报错
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))
    # 模型输出文件，如果不存在则创建
    tf.gfile.MakeDirs(FLAGS.output_dir)
    task_name = FLAGS.task_name.lower()  # 从命令行中获取任务名
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    # 根据任务字典选择对应Processor，这里是news项目，所以选NewsProcessor
    processor = processors[task_name]()
    label_list = processor.get_labels()  # 获取项目结果标签集 [0,1]

    # FullTokenizer是聚合分词类，它以vocab_file为词典，将词转化为该词对应的id，
    # 对于某些特殊词，如johanson，会先将johanson按照最大长度拆分，
    # 再看拆分的部分是否在vocab_file里。vocab_file里有没有"johanson"这个词，
    # 但有"johan"和"##son"这两个词，所以将"johanson"这个词拆分成两个词
    # （##表示非开头匹配）
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    # 无tpu忽略
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    """1.训练配置开始"""
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)  # 一行行读取训练集train.tsv
        # 计算总batch数量：（总数据量/一个batch的大小）*训练次数
        num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        # warmup目的是让前面的batch的学习率降低，后面的batch的学习率恢复之前设置
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # 模型构建初始化，返回的是一个方法
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # 如果没有TPU，就会用回退到CPU或GPU的Estimator
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,  # 是否使用tpu
        model_fn=model_fn,  # 模型构建方法
        config=run_config,  #
        train_batch_size=FLAGS.train_batch_size,  # 训练数据batch的大小
        eval_batch_size=FLAGS.eval_batch_size,  # 评估数据batch的大小
        predict_batch_size=FLAGS.predict_batch_size)  # 预测数据batch的大小

    if FLAGS.do_train:
        # 训练
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")  # 训练存放结果文件
        # 数据读取核心：将训练数据集加载写到TFRecord，训练时模型读取TFRecord的数据会速度更快
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        # 返回file_based_input_fn_builder.input_fn方法，训练数据集提取
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        # 模型构建+训练
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        # 评估
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)  # 获取dev数据集
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        # 预测
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
    # 读取命令行参数
    flags.mark_flag_as_required("data_dir")  # 数据集位置
    flags.mark_flag_as_required("task_name")  # 任务名
    flags.mark_flag_as_required("vocab_file")  # 所有出现的词汇总文件
    flags.mark_flag_as_required("bert_config_file")  # bert模型需要的配置参数
    flags.mark_flag_as_required("output_dir")  # 输出文件路径
    tf.app.run()
