from components.transformer import transformer
import tensorflow_datasets as tfds
import tensorflow as tf

START_TOKEN = 0
END_TOKEN = 20
MAX_LENGTH = 10

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    ["test vova test", "corpus"], target_vocab_size=2 ** 13)


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


def print_hi(name):
    sample_transformer = transformer(
        vocab_size=8192,
        num_layers=6,
        units=512,
        d_model=256,
        num_heads=8,
        dropout=0.3,
        is_encoder=True,
        name="sample_transformer")

    # pad tokenized sentences
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

        sample_transformer([inputs, dec_inputs])
        break

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
