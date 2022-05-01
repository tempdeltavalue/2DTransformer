import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, add, LayerNormalization, Embedding, Dropout, Input

from components.masks import create_padding_mask, create_look_ahead_mask
from components.decoder import decoder
from components.encoder import encoder

def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                is_encoder=False,
                name="transformer"):

    inputs = Input(shape=(None,), name="inputs")

    dec_inputs = Input(shape=(None,), name="dec_inputs")

    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = Lambda(create_look_ahead_mask,
                             output_shape=(1, None, None),
                             name='look_ahead_mask')(dec_inputs)

    # mask the encoder outputs for the 2nd attention block
    dec_padding_mask = Lambda(create_padding_mask,
                              output_shape=(1, 1, None),
                              name='dec_padding_mask')(inputs)

    if is_encoder is True:
        enc_padding_mask = Lambda(
            create_padding_mask, output_shape=(1, 1, None),
            name='enc_padding_mask')(inputs)

        enc_outputs = encoder(
            vocab_size=vocab_size,
            num_layers=num_layers,
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )(inputs=[inputs, enc_padding_mask])

        decoder_input = [dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask]
    else:
        embeddings = Embedding(vocab_size, d_model)(dec_inputs)
        #
        # print("Embedding shape", embeddings.shape)
        # # embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

        decoder_input = [dec_inputs, embeddings, look_ahead_mask, dec_padding_mask]

    dec_outputs = decoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
    )(inputs=decoder_input) #dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask

    outputs = Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)