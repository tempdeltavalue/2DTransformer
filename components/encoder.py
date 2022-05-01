import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, add, LayerNormalization, Embedding, Dropout
from components.multihead_attention import MultiHeadAttention
from components.positional_encoding import PositionalEncoding



def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  attention = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': padding_mask
      })
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

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)

if __name__ == "__main__":
    sample_encoder = encoder(
        vocab_size=8192,
        num_layers=2,
        units=512,
        d_model=128,
        num_heads=4,
        dropout=0.3,
        name="sample_encoder")