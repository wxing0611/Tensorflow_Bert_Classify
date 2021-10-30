# coding=utf-8
"""
    Bert模型与相关功能
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf


class BertConfig(object):
    """Bert配置文件类"""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
          hidden_size: Size of the encoder layers and the pooler layer.
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
          intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
          hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
          attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
          max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
          type_vocab_size: The vocabulary size of the `token_type_ids` passed into
            `BertModel`.
          initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """ 封装成字典对象 """
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """ 读取json配置文件 """
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """ Serializes this instance to a Python dictionary. """
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """ Serializes this instance to a JSON string. """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertModel(object):
    """BERT模型构建类
    Example usage:

    ```python
    # Already been converted into WordPiece token ids
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
      num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config, is_training=True,
      input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

    label_embeddings = tf.get_variable(...)
    pooled_output = model.get_pooled_output()
    logits = tf.matmul(pooled_output, label_embeddings)
    ...
    ```
    """

    def __init__(self,
                 config,  # bert配置文件信息
                 is_training,  # 是否训练，True/False
                 input_ids,  # 输入词id列表（batch_size,seq_length）
                 input_mask=None,   # 填充词标记列表（batch_size,seq_length）
                 token_type_ids=None,   # 句子序号列表（batch_size,seq_length）
                 use_one_hot_embeddings=False,
                 scope=None):
        """
            bert模型构造
        :param config: bert配置文件信息
        :param is_training: 是否训练，True/False
        :param input_ids: 输入词id列表（batch_size,seq_length）
        :param input_mask: 填充词标记列表（batch_size,seq_length）
        :param token_type_ids: 句子序号列表（batch_size,seq_length）
        :param use_one_hot_embeddings:
        :param scope:
        """
        config = copy.deepcopy(config)
        if not is_training:
            # 如果是评估eval，则修改下面两个参数
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
        # 读取数据维度对应的数值
        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0] # 8
        seq_length = input_shape[1] # 128

        # 如果前面没有初始化，下面你会再次初始化
        if input_mask is None:  # 如果没设置mask 自然就都是1的，填充字符也会被设置为1
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:  # 默认只有一句话，全为0标志
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.variable_scope(scope, default_name="bert"):
            with tf.variable_scope("embeddings"):
                # embedding层搭建
                # 将输入矩阵input_ids[batch_size, seq_length] => embedding_output词向量矩阵[batch_size, seq_length, embedding_size],
                # embedding_table总词向量矩阵命名embedding_table [vocab_size, embedding_size]
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=input_ids,  # 每批输入格式 （8,128）
                    vocab_size=config.vocab_size,  # 总词数
                    embedding_size=config.hidden_size,  # embedding维度
                    initializer_range=config.initializer_range,  # 初始值范围
                    word_embedding_name="word_embeddings",
                    use_one_hot_embeddings=use_one_hot_embeddings)  # tpu才用到use_one_hot_embeddings

                # output: 将embedding输出 + 句子序号编码向量 + 词位置编码向量， 并对output进行layernorm + dropout操作
                # embedding_output : [batch_size, seq_length, embedding_size]
                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,  # 传入上面embedding的输出 [batch_size, seq_length, embedding_size]
                    use_token_type=True,
                    token_type_ids=token_type_ids,  # 句子序号列表 [batch_size, seq_length]
                    token_type_vocab_size=config.type_vocab_size,  # 总去重词数 vocab_size
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    initializer_range=config.initializer_range,   # 初始值范围
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)

            with tf.variable_scope("encoder"):
                # encoder层搭建  【bert模型构建】
                # [batch_size, seq_length] => [batch_size, seq_length, seq_length]
                # 第三维度seq_length的每个词与本句哪些词相关，第三维度就是来标记每个词与本句哪些词要参与计算（0,1标记，填充词不参与计算）
                # 由于有些句子不够seq_length个词，需要用0填充。填充的位置我们不参与 self-attention 运算
                attention_mask = create_attention_mask_from_input_mask(input_ids, input_mask)

                # 构建transformer模型  [batch_size, seq_length, hidden_size].
                self.all_encoder_layers = transformer_model(
                    input_tensor=self.embedding_output,  # embedding的输出作为encoder的输入 [batch_size, seq_length, embedding_size]
                    attention_mask=attention_mask,  # self-attention参与计算的词 [batch_size, seq_length, seq_length]
                    hidden_size=config.hidden_size,   # 隐藏层数量，也是输出数量
                    num_hidden_layers=config.num_hidden_layers,  # Transformer中的隐层层数
                    num_attention_heads=config.num_attention_heads,  # 多少组head（类似CNN的filter）， 一组head包含Wq,Wk,Wv
                    intermediate_size=config.intermediate_size,  # 全连接层神经元个数
                    intermediate_act_fn=get_activation(config.hidden_act),  # 激活函数，默认relu
                    hidden_dropout_prob=config.hidden_dropout_prob,  # 隐层dropout
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob, # attention dropout
                    initializer_range=config.initializer_range,  # 初始值范围
                    do_return_all_layers=True)  # 是否返回每一层的输出

            # 获取最后一层 构建全连接类别输出
            self.sequence_output = self.all_encoder_layers[-1]  # [batch, seq_length, embedding_size]
            with tf.variable_scope("pooler"):
                # 找到第一个单元【CLS单元】 [batch, embedding_size]
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                # 做全连接，返回hidden_size
                self.pooled_output = tf.layers.dense(
                    first_token_tensor,
                    config.hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=create_initializer(config.initializer_range))

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        """Gets final hidden layer of encoder.

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the final hidden of the transformer encoder.
        """
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        """Gets output of the embedding lookup (i.e., input to the transformer).

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the output of the embedding layer, after summing the word
          embeddings with the positional embeddings and the token type embeddings,
          then performing layer normalization. This is the input to the transformer.
        """
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def get_activation(activation_string):
    """获取激活函数"""
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """
        Compute the union of the current variables and checkpoint variables.
    :param tvars:
    :param init_checkpoint: 模型参数路径
    :return:
    """
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    """
        Runs layer normalization followed by dropout.
    :param input_tensor: embedding输出 + 句子序号编码向量 + 词位置编码向量
    :param dropout_prob: dropout占比
    :param name:
    :return:
    """
    output_tensor = layer_norm(input_tensor, name)  # 调用TensorFlow的layernorm函数
    output_tensor = dropout(output_tensor, dropout_prob)  # 进行dropout
    return output_tensor


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
    """

    :param input_ids: 每批输入格式 （8,128）
    :param vocab_size: 总词数
    :param embedding_size: embedding维度数
    :param initializer_range: 初始值范围
    :param word_embedding_name:
    :param use_one_hot_embeddings:
    :return:  [batch_size, seq_length, embedding_size], [vocab_size, embedding_size]
    """
    # [batch_size, seq_length] --> [batch_size, seq_length, 1]
    if input_ids.shape.ndims == 2:
        # 维度2维，添加1维，变3维
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    # 初始化embedding矩阵,并设定初始值 [vocab_size, embedding_size]
    embedding_table = tf.get_variable(  # 词映射矩阵，30522, 768
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range))

    flat_input_ids = tf.reshape(input_ids, [-1])  # [batch_size*seq_length]
    if use_one_hot_embeddings:
        # tpu会采用这里
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        # [batch_size*seq_length] => [batch_size*seq_length, embedding_size]
        output = tf.gather(embedding_table, flat_input_ids)  # CPU,GPU运算(1024, 768) 一个batch里所有的映射结果，这里的1024=8*128

    input_shape = get_shape_list(input_ids)  # 获取数据各维度长度值
    # [batch_size*seq_length, embedding_size] ==> [batch_size, seq_length, embedding_size]
    output = tf.reshape(output,
                        input_shape[0:-1] + [input_shape[-1] * embedding_size])  # (8, 128, 768)
    return (output, embedding_table)


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
    """
        output: 将embedding输出 + 句子序号编码向量 + 词位置编码向量， 并对output进行layernorm + dropout操作
    :param input_tensor: 传入上面embedding的输出 [batch_size, seq_length, embedding_size]
    :param use_token_type:
    :param token_type_ids:  句子序号列表 [batch_size, seq_length] [[0,0,..,0],[1,1,..,1]]
    :param token_type_vocab_size:   句子数量 2
    :param token_type_embedding_name:  句子序号embedding名
    :param use_position_embeddings:  是否添加位置embedding
    :param position_embedding_name:  位置embedding名
    :param initializer_range:   初始化值范围
    :param max_position_embeddings:   最大的seq_len限制
    :param dropout_prob:
    :return: [batch_size, seq_length, embedding_size]
    """
    input_shape = get_shape_list(input_tensor, expected_rank=3) # 获取三个维度长度值
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]  # embedding_size

    output = input_tensor

    if use_token_type:
        # 判断是否添加句子编码
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if"
                             "`use_token_type` is True.")
        # 初始化句子序号embedding    (2, embedding_size)
        # 记录是第一句话还是第二句话的向量
        token_type_table = tf.get_variable(  #
            name=token_type_embedding_name,
            shape=[token_type_vocab_size, width],
            initializer=create_initializer(initializer_range))
        # [batch_size, seq_length] ==> [batch_size*seq_length]
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])  # 8*128 = 共有1024个词
        # [batch_size*seq_length] ==> [batch_size*seq_length, 句子数量]
        # 对此进行one-hot编码 (1024,2)  由于案例输入的两个句子，(0,1)表示该词属于第一个句子，(1,0)表示该词属于第二个句子
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
        # (batch_size*seq_length,2)*(2,embedding_size) = (batch_size*seq_length,embedding_size)
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)  # (1024,2)*(2,768) = (1024,768)
        # (batch_size*seq_length,embedding_size) => (batch_size, seq_length, embedding_size)
        token_type_embeddings = tf.reshape(token_type_embeddings, [batch_size, seq_length, width]) # 格式转换 8, 128, 768
        output += token_type_embeddings  # 将第一句与第二句位置的embedding编码融入到输入词向量中。

    if use_position_embeddings:
        # 判断是否要做词位置编码信息
        # 句子长度seq_length需要小于等于max_position_embeddings
        assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            # 初始化填充位置embeddings矩阵 [max_position_embeddings, embedding_size]
            full_position_embeddings = tf.get_variable(
                name=position_embedding_name,
                shape=[max_position_embeddings, width],
                initializer=create_initializer(initializer_range))

            # [max_position_embeddings, embedding_size] => [seq_length, embedding_size]
            position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                           [seq_length, -1])
            num_dims = len(output.shape.as_list()) # 维度数=2

            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend(
                [seq_length, width])  # [1, 128, 768] 表示位置编码跟输入啥数据无关，因为原始的embedding是有batchsize当做第一个维度，这里为了计算也得加入
            # [seq_length, embedding_size] => [1, seq_length, embedding_size]
            position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)
            output += position_embeddings

    output = layer_norm_and_dropout(output, dropout_prob)  # 进行归一化与dropout
    return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """
        输入[batch_size, seq_length]，输出[batch_size, seq_length, seq_length]
        第三维度就是来标记每个词与本句哪些词要参与计算（0,1标记，填充词不参与计算）
    :param from_tensor: 输入词id列表  [batch_size,seq_length]
    :param to_mask: 填充词标记列表  [batch_size,seq_length]
    :return: [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    # [batch_size, to_seq_length] => [batch_size, 1, to_seq_length]
    to_mask = tf.cast(tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # [batch_size, from_seq_length, 1] * [batch_size, 1, to_seq_length] = [batch_size, from_seq_length, to_seq_length]
    mask = broadcast_ones * to_mask

    return mask


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
    """
        构建self-attention层
    :param from_tensor:  输入句子 [batch_size*seq_length, embedding_size]
    :param to_tensor:  关联句子：也是输入句子 [batch_size*seq_length, embedding_size]
    :param attention_mask:  每句话中的每个词与本句哪些词参与计算（填充词不参与self-attention计算） [batch_size, seq_length, seq_length]
    :param num_attention_heads:  head组数
    :param size_per_head:  每个head需要返回的维度数
    :param query_act: q矩阵激活函数
    :param key_act: k矩阵激活函数
    :param value_act: v矩阵激活函数
    :param attention_probs_dropout_prob:  attention-dropout
    :param initializer_range:  初始值范围
    :param do_return_2d_tensor: 3d拉成2d,可能计算更快
    :param batch_size: batch
    :param from_seq_length: 输入句子长度
    :param to_seq_length: 关联句子长度
    :return:
    """
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads, seq_length, width):
        """
            维度转换与切换
        :param input_tensor: 输入向量
        :param batch_size: batch大小
        :param num_attention_heads: head组数量
        :param seq_length: 句子长度
        :param width: 每个head需要返回的维度数
        :return: [batch_size, heads组数, seq_length, head返回维度]
        """
        output_tensor = tf.reshape(input_tensor, [batch_size, seq_length, num_attention_heads, width])
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])   # 返回维度数量: [1024, 768]
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])   # 返回维度数量: [1024, 768]

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")
    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]

    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences) 8
    #   F = `from_tensor` sequence length 128  从哪个句子抽词出来
    #   T = `to_tensor` sequence length 128  去那个句子词中计算
    #   N = `num_attention_heads` 12   多少个head
    #   H = `size_per_head` 64    每个head输出维度多少维
    #   输出维度是768 = 12 * 64

    #  如果正常，传入的就是2维的数据。
    from_tensor_2d = reshape_to_matrix(from_tensor)  # (1024, 768) [batch_size*seq_length, embedding_size]
    to_tensor_2d = reshape_to_matrix(to_tensor)
    # 构建Wq全连接(embedding_size -> num_attention_heads * size_per_head)
    # `query_layer` = [batch size*from_tensor, num_attention_heads*size_per_head]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range))
    # 构建Wk全连接(embedding_size -> num_attention_heads * size_per_head)
    # `key_layer` = [batch size*to_tensor, num_attention_heads*size_per_head]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=create_initializer(initializer_range))
    # 构建Wv全连接(embedding_size -> num_attention_heads * size_per_head)
    # `value_layer` = [batch size*to_tensor, num_attention_heads*size_per_head]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=create_initializer(initializer_range))

    # 对数据进行格式转换 `query_layer` = [batch size, num_attention_heads, from_tensor, size_per_head]  为了加速计算内积 (8, 12, 128, 64)
    query_layer = transpose_for_scores(query_layer, batch_size, num_attention_heads, from_seq_length, size_per_head)

    # 对数据进行格式转换 `key_layer` = [batch size, num_attention_heads, to_tensor, size_per_head]  为了加速计算内积 (8, 12, 128, 64)
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads, to_seq_length, size_per_head)

    # Wq*Wk 计算attention值  [batch size, num_attention_heads, from_tensor, to_tensor]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)  # 结果为(8, 12, 128, 128)
    attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))  # 消除维度对结果的影响

    if attention_mask is not None:
        # 只计算句子中原词,填充词不计算,下面将填充词忽略其影响结果。
        # `attention_mask` = [batch_size, seq_length, seq_length] -> [batch_size, 1, from_tensor, to_tensor]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])
        # mask为1的时候结果为0 mask为0的时候结果为非常大的负数,这样计算softmax,非常大的负数结果也是非常小的，从而达到忽略填充值计算
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
        # 把这个加入到原始的得分里相当于mask为1的就不变，mask为0的就会变成非常大的负数
        attention_scores += adder

    # softmax计算：`attention_probs` = [batch size, num_attention_heads, from_tensor, to_tensor]
    attention_probs = tf.nn.softmax(attention_scores)  # 再做softmax此时负数做softmax相当于结果为0了就相当于不考虑了
    # dropout计算,得到最后每个词与本句每个词的影响程度权重值  [batch size, num_attention_heads, from_tensor, to_tensor]
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)
    # `value_layer` = [batch size*to_tensor, num_attention_heads*size_per_head] -> [batch size, to_tensor, num_attention_heads, size_per_head]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])  # (8, 128, 12, 64)
    # 维度转换 `value_layer` = [batch size, num_attention_heads, to_tensor, size_per_head]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])  # (8, 12, 128, 64)
    # [batch size, num_attention_heads, from_tensor, to_tensor] * [batch size, num_attention_heads, to_tensor, size_per_head]
    context_layer = tf.matmul(attention_probs, value_layer)  # 计算最终结果特征 (8, 12, 128, 64)  [batch size, num_attention_heads, from_tensor, size_per_head]
    # 维度转换 `context_layer` = [batch size, from_tensor, num_attention_heads, size_per_head]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])  # 转换回[8, 128, 12, 64]
    if do_return_2d_tensor:
        # `context_layer` = [batch size, from_tensor, num_attention_heads, size_per_head] -> [batch size*from_tensor, num_attention_heads*size_per_head]
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [batch size, from_tensor, num_attention_heads*size_per_head]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])  # (1024, 768)

    return context_layer  # (1024, 768) [batch size*from_tensor, num_attention_heads*size_per_head]


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
    """
        transformer 模型构建
    :param input_tensor: embedding的输出作为encoder的输入 [batch_size, seq_length, embedding_size]
    :param attention_mask: 每句话中的每个词与本句哪些词参与计算（填充词不参与self-attention计算） [batch_size, seq_length, seq_length]
    :param hidden_size:  隐藏层数量，也是输出数量。 输入与输出维度需要一样，便于后面残差构建
    :param num_hidden_layers:  Transformer中的隐层层数
    :param num_attention_heads:  多少组head（类似CNN的filter）默认12组， 一组head包含Wq,Wk,Wv
    :param intermediate_size:  全连接层神经元个数
    :param intermediate_act_fn:  激活函数，默认relu
    :param hidden_dropout_prob:  隐层dropout
    :param attention_probs_dropout_prob:  attention dropout
    :param initializer_range: 初始值范围
    :param do_return_all_layers: 是否返回每一层的输出
    :return:
    """
    if hidden_size % num_attention_heads != 0:
        # num_attention_heads 则是原理一组Wq,Wk,Wv为一个head，有num_attention_heads个head
        # 输入经过一个head1得到输出1， 经过head2得到输出2，将所有输出concat等于hidden_size，所以需要hidden_size % num_attention_heads能整除
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)  # 假设一共要输出hidden_size是768个特征，给每个head分一下，每个head需要返回的维度数
    input_shape = get_shape_list(input_tensor, expected_rank=3)  # 获取三个维度数量，[8, 128, 768]
    batch_size = input_shape[0]  # batch_size
    seq_length = input_shape[1]  # seq_length
    input_width = input_shape[2]  # embedding_size

    if input_width != hidden_size:
        # 注意残差连接的方式，需要它俩维度一样才能相加。
        # 这里残差连接达到忽略某self-attention层的输出就让改成权重参数为0输出结果与self-attention层的输入进行累加
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" % (input_width, hidden_size))

    # 维度变换： [batch_size,seq_length,embedding_size] => [batch_size*seq_length, embedding_size]
    prev_output = reshape_to_matrix(input_tensor)  # 这里目的可能是为了加速，照写即可

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        # 循环搭建隐藏层
        with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output  # 上一层的输出，该层的输入 [batch_size*seq_length, embedding_size]

            # 下面是self-attention层
            with tf.variable_scope("attention"):
                attention_heads = []
                with tf.variable_scope("self"):
                    # 构建self-attention层
                    # from_tensor与to_tensor指定是同一个输入，就是表示self-attention，本句的词与本句进行attention计算
                    # attention_head = [batch size*from_tensor, num_attention_heads*size_per_head]
                    attention_head = attention_layer(
                        from_tensor=layer_input,   # 输入句子 [batch_size*seq_length, embedding_size]
                        to_tensor=layer_input,  # 关联句子 [batch_size*seq_length, embedding_size]
                        attention_mask=attention_mask,  # 每句话中的每个词与本句哪些词参与计算（填充词不参与self-attention计算） [batch_size, seq_length, seq_length]
                        num_attention_heads=num_attention_heads,  # head组数
                        size_per_head=attention_head_size,  # 每个head需要返回的维度数
                        attention_probs_dropout_prob=attention_probs_dropout_prob,  # attention-dropout
                        initializer_range=initializer_range,  # 初始值范围
                        do_return_2d_tensor=True,  # 3d拉成2d,可能计算更快
                        batch_size=batch_size,  # batch
                        from_seq_length=seq_length,  # 句子长度
                        to_seq_length=seq_length)  # 句子长度
                    attention_heads.append(attention_head)

                # attention输出
                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    # 如果多个attention输出,就将多个attention的输出进行concat操作.
                    attention_output = tf.concat(attention_heads, axis=-1)

                # 全连接层 + dropout + layerNorm
                with tf.variable_scope("output"):
                    # 1024, 768 残差连接
                    attention_output = tf.layers.dense(
                        attention_output,  # [batch size*from_tensor, num_attention_heads*size_per_head]
                        hidden_size,
                        kernel_initializer=create_initializer(initializer_range))
                    attention_output = dropout(attention_output, hidden_dropout_prob)
                    attention_output = layer_norm(attention_output + layer_input)  # layerNorm归一化

            # attention层外部的全连接层1
            with tf.variable_scope("intermediate"):  # 全连接层 (768, 3072)
                intermediate_output = tf.layers.dense(
                    attention_output, intermediate_size, activation=intermediate_act_fn,  # 激励函数
                    kernel_initializer=create_initializer(initializer_range))
            # attention层外部的全连接层2
            # Down-project back to `hidden_size` then add the residual.
            with tf.variable_scope("output"):  # 再变回一致的维度
                layer_output = tf.layers.dense(
                    intermediate_output,
                    hidden_size,
                    kernel_initializer=create_initializer(initializer_range))
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)  # layerNorm归一化
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

    # 是否返回所有层输出
    if do_return_all_layers:
        # 所有layer层的输出
        final_outputs = []
        for layer_output in all_layer_outputs:
            # [batch_size*seq_length, embedding_size] => [batch_size, seq_length, embedding_size]
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output) # [num_hidden_layers, batch_size, seq_length, embedding_size]
        return final_outputs
    else:
        # 最后一层输出 [batch_size, seq_length, embedding_size]
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """
        改变格式
    :param input_tensor: [batch_size,seq_length,embedding_size]
    :return: [batch_size*seq_length,embedding_size]
    """
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
