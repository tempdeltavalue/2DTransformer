from components.decoder import decoder

import tensorflow as tf

from tensorflow.keras.layers import Dense, Lambda, add, LayerNormalization, Embedding, Dropout, Input, TimeDistributed, LSTM, Attention
import tensorflow_datasets as tfds
from components.positional_encoding import PositionalEncoding
from components.masks import create_padding_mask




tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    ["test vova test", "corpus"], target_vocab_size=2 ** 13)

MAX_LENGTH = 10

def tokenize_and_filter(inputs, outputs):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        # tokenize sentence
        print("sksk", sentence1)
        print("sksk", tokenizer.encode(sentence1))
        sentence1 = tokenizer.encode(sentence1)
        sentence2 = tokenizer.encode(sentence2)
        # check tokenized sentence max length
        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

    # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

    return tokenized_inputs, tokenized_outputs

# sample_encoder = encoder(
#     vocab_size=8192,
#     num_layers=2,
#     units=512,
#     d_model=128,
#     num_heads=4,
#     dropout=0.3,
#     name="sample_encoder")

dec_outputs = decoder(
    vocab_size=8192,
    num_layers=2,
    units=512,
    d_model=128,
    num_heads=4,
    dropout=0.3,
)























def scaled_dot_product_attention(query, key, value, mask):
    """Calculate the attention weights. """
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to zero out padding tokens
    if mask is not None:
        logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output

class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = Dense(units=d_model)
        self.key_dense = Dense(units=d_model)
        self.value_dense = Dense(units=d_model)

        self.dense = Dense(units=d_model)

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'd_model': self.d_model,
        })
        return config

    def split_heads(self, inputs, batch_size):
        print("\n")
        print("inp shape", inputs.shape)

        in_shape = (batch_size, -1, self.num_heads, self.depth)
        print("res shape", in_shape)
        inputs = Lambda(lambda inputs: tf.reshape(inputs, shape=in_shape))(inputs)

        return Lambda(lambda inputs: tf.transpose(inputs, perm=[0, 2, 1, 3]))(inputs)

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention

        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = Lambda(lambda scaled_attention: tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]))(scaled_attention)

        # concatenation of heads
        concat_attention = Lambda(lambda scaled_attention: tf.reshape(scaled_attention,
                                                                      (batch_size, -1, self.d_model)))(scaled_attention)

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs











def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")


  attention = Attention() ([inputs, padding_mask])

      #
      # MultiHeadAttention(
      # d_model, num_heads, name="attention")({
      #     'query': inputs,
      #     'key': inputs,
      #     'value': inputs,
      #     'mask': padding_mask
      # })
  attention = Dropout(rate=dropout)(attention)
  add_attention = add([inputs,attention])
  attention = LayerNormalization(epsilon=1e-6)(add_attention)

  outputs = Dense(units=units, activation='relu')(attention)
  outputs = Dense(units=d_model)(outputs)
  outputs = Dropout(rate=dropout)(outputs)
  add_attention = add([attention,outputs])
  outputs = LayerNormalization(epsilon=1e-6)(add_attention)

  return tf.keras.Model(inputs=[inputs, padding_mask],
                        outputs=outputs,
                        name=name)


def encoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):

          inputs = tf.keras.Input(shape=(None,), name="inputs")
          padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

          embeddings = Embedding(vocab_size, d_model)(inputs)
          embeddings *= Lambda(lambda d_model: tf.math.sqrt(tf.cast(d_model, tf.float32)))(d_model)
          embeddings = PositionalEncoding(vocab_size,d_model)(embeddings)

          outputs = Dropout(rate=dropout)(embeddings)

          for i in range(num_layers):
            outputs = encoder_layer(
                units=units,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                name="encoder_layer_{}".format(i),
            )([outputs, padding_mask])

          return tf.keras.Model(inputs=[inputs, padding_mask],
                                outputs=outputs, name=name)


vocab_size = 8128
num_layers = 2
units = 64
d_model = 128
num_heads = 8
dropout = 0.1

encoder = encoder(vocab_size,
                    num_layers,
                    units,
                    d_model,
                    num_heads,
                    dropout,)

inputs = Input(shape=(None, ), name="inputs")
enc_padding_mask = Lambda(
    create_padding_mask, output_shape=(1, 1, None),
    name='enc_padding_mask')(inputs)

enc_outputs = encoder (inputs=[inputs, enc_padding_mask])
# print(enc_outputs.input)
x = LSTM(units=128) (enc_outputs)
#
# x = dec_outputs(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask]) #  enc_outputs -> x
#
# model = tf.keras.Model(inputs=enc_outputs.input, outputs=enc_outputs.output)

questions, answers = tokenize_and_filter(["test vova", "test 2"],
                                         ["corpus", "corpus2"])

dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]
    },
    {
        'outputs': answers[:, 1:]
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(10)
dataset = dataset.batch(10)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

for item in dataset:
    inputs = item[0]["inputs"]
    dec_inputs = item[0]["dec_inputs"]

    print("inputs", inputs.shape)
    print("dec_inputs", dec_inputs.shape)

    model([inputs, dec_inputs])
    break