# %%
"""
Create a Hinglish translation from English text. The text should sound natural and also
convert all the difficult words and phrases in English to Hinglish. This converted text should
be easy to understand for even a non-native Hindi speaker.
We have attached below the statements that are required to be used for this assignment.
1. Definitely share your feedback in the comment section.
2. So even if it's a big video, I will clearly mention all the products.
3. I was waiting for my bag.
Example:
Statement: I had about a 30 minute demo just using this new headset
Output required: मझु ेसि र्फ ३० minute का demo मि ला था इस नयेheadset का इस्तमे ाल करनेके
लि ए
Rules:
● The model must be able to generate a translation that is indistinguishable from
Hindi spoken by a casual Hindi speaker.
● Must be able to keep certain words in English to keep the Hindi translation Easy.
● The Hinglish sentences should be accurate to the meaning of the original sentence
"""

# use MT5 model for this
# use pretrained model
# use hinglish data for training

# %%
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
)
from tensorflow.keras.models import (
    Model,
)
from tensorflow.keras.preprocessing.text import (
    Tokenizer,
)
from tensorflow.keras.preprocessing.sequence import (
    pad_sequences,
)
import numpy as np
import nltk.translate.bleu_score as bleu
import random
import string
from sklearn.model_selection import (
    train_test_split,
)
import os
import time

# %%
dataset = pd.read_csv(
    "data/synthetic-dataset/train.csv"
)
dataset.head()

# %%
dataset.describe()

# %%
dataset

# %%
# create a new dataframe of english and hinglish column
df = (
    pd.DataFrame()
)
df[
    "english"
] = dataset[
    "English"
]
df[
    "hinglish"
] = dataset[
    "Hinglish"
]
df.head()

# %%
exclude = set(
    string.punctuation
)  # Set of all special characters
remove_digits = str.maketrans(
    "",
    "",
    string.digits,
)  # Set of all digits


# %%
# write function to preprocess english sentence
def preprocess_english_sentence(
    sentence,
):
    sentence = (
        sentence.lower()
    )
    sentence = re.sub(
        "'",
        "",
        sentence,
    )
    sentence = "".join(
        ch
        for ch in sentence
        if ch
        not in exclude
    )
    sentence = sentence.translate(
        remove_digits
    )
    sentence = (
        sentence.strip()
    )
    sentence = re.sub(
        " +",
        " ",
        sentence,
    )
    sentence = (
        "<start> "
        + sentence
        + " <end>"
    )
    return sentence


# %%
# preprocess hinglish sentense
def preprocess_hinglish_sentence(
    sentence,
):
    sentence = (
        sentence.lower()
    )
    sentence = re.sub(
        "'",
        "",
        sentence,
    )
    sentence = "".join(
        ch
        for ch in sentence
        if ch
        not in exclude
    )
    sentence = sentence.translate(
        remove_digits
    )
    sentence = (
        sentence.strip()
    )
    sentence = re.sub(
        " +",
        " ",
        sentence,
    )
    sentence = (
        "<start> "
        + sentence
        + " <end>"
    )
    return sentence


# %%
df[
    "english"
] = df[
    "english"
].apply(
    preprocess_english_sentence
)
df[
    "hinglish"
] = df[
    "hinglish"
].apply(
    preprocess_hinglish_sentence
)

df.rename(
    columns={
        "english_sentence": "english",
        "hindi_sentence": "hindi",
    },
    inplace=True,
)

df.head()


# %%
# tokenzizer
def tokenizer(
    language,
):
    tokenizer = Tokenizer(
        filters="",
        split=" ",
    )
    tokenizer.fit_on_texts(
        language
    )
    tensor = tokenizer.texts_to_sequences(
        language
    )
    tensor = pad_sequences(
        tensor,
        padding="post",
    )
    return (
        tensor,
        tokenizer,
    )


# %%
def load_dataset():
    (
        input_tensor,
        inp_lang_tokenizer,
    ) = tokenizer(
        df[
            "english"
        ].values
    )
    (
        target_tensor,
        targ_lang_tokenizer,
    ) = tokenizer(
        df[
            "hinglish"
        ].values
    )
    return (
        input_tensor,
        target_tensor,
        inp_lang_tokenizer,
        targ_lang_tokenizer,
    )


# %%
(
    input_tensor,
    target_tensor,
    input_lang,
    target_lang,
) = (
    load_dataset()
)

# %%
(
    max_length_targ,
    max_length_inp,
) = (
    target_tensor.shape[
        1
    ],
    input_tensor.shape[
        1
    ],
)


# %%

(
    input_tensor_train,
    input_tensor_val,
    target_tensor_train,
    target_tensor_val,
) = train_test_split(
    input_tensor,
    target_tensor,
    test_size=0.2,
)

print(
    len(
        input_tensor_train
    ),
    len(
        target_tensor_train
    ),
    len(
        input_tensor_val
    ),
    len(
        target_tensor_val
    ),
)

# %%
BUFFER_SIZE = len(
    input_tensor_train
)
BATCH_SIZE = (
    64
)
N_BATCH = (
    BUFFER_SIZE
    // BATCH_SIZE
)
embedding_dim = (
    256
)
units = (
    1024
)
steps_per_epoch = (
    len(
        input_tensor_train
    )
    // BATCH_SIZE
)

vocab_inp_size = len(
    input_lang.word_index.keys()
)
vocab_tar_size = len(
    target_lang.word_index.keys()
)

dataset = tf.data.Dataset.from_tensor_slices(
    (
        input_tensor_train,
        target_tensor_train,
    )
).shuffle(
    BUFFER_SIZE
)
dataset = dataset.batch(
    BATCH_SIZE,
    drop_remainder=True,
)

# %%
embeddings_index = (
    dict()
)
f = open(
    "glove-2.txt",
    "r+",
)
for (
    line
) in f:
    values = (
        line.split()
    )
    word = values[
        0
    ]
    coefs = np.asarray(
        values[
            1:
        ],
        dtype="float32",
    )
    embeddings_index[
        word
    ] = coefs
f.close()

embedding_matrix = np.zeros(
    (
        vocab_inp_size
        + 1,
        300,
    )
)
for (
    word,
    i,
) in (
    input_lang.word_index.items()
):
    embedding_vector = embeddings_index.get(
        word
    )
    if (
        embedding_vector
        is not None
    ):
        embedding_matrix[
            i
        ] = embedding_vector


# %%
class Encoder(
    tf.keras.Model
):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        enc_units,
        batch_sz,
    ):
        super(
            Encoder,
            self,
        ).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name="embedding_layer_encoder",
            trainable=False,
        )
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_activation="sigmoid",
            recurrent_initializer="glorot_uniform",
        )

    def call(
        self,
        x,
        hidden,
    ):
        x = self.embedding(
            x
        )
        (
            output,
            state,
        ) = self.gru(
            x,
            initial_state=hidden,
        )
        return (
            output,
            state,
        )

    def initialize_hidden_state(
        self,
    ):
        return tf.zeros(
            (
                self.batch_sz,
                self.enc_units,
            )
        )


# %%
class Decoder(
    tf.keras.Model
):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        dec_units,
        batch_sz,
    ):
        super(
            Decoder,
            self,
        ).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
        )
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_activation="sigmoid",
            recurrent_initializer="glorot_uniform",
        )
        self.fc = tf.keras.layers.Dense(
            vocab_size
        )

        # used for attention
        self.W1 = tf.keras.layers.Dense(
            self.dec_units
        )
        self.W2 = tf.keras.layers.Dense(
            self.dec_units
        )
        self.V = tf.keras.layers.Dense(
            1
        )

    def call(
        self,
        x,
        hidden,
        enc_output,
    ):
        hidden_with_time_axis = tf.expand_dims(
            hidden,
            1,
        )

        score = self.V(
            tf.nn.tanh(
                self.W1(
                    enc_output
                )
                + self.W2(
                    hidden_with_time_axis
                )
            )
        )

        attention_weights = tf.nn.softmax(
            score,
            axis=1,
        )

        context_vector = (
            attention_weights
            * enc_output
        )
        context_vector = tf.reduce_sum(
            context_vector,
            axis=1,
        )

        x = self.embedding(
            x
        )

        x = tf.concat(
            [
                tf.expand_dims(
                    context_vector,
                    1,
                ),
                x,
            ],
            axis=-1,
        )

        (
            output,
            state,
        ) = self.gru(
            x
        )

        output = tf.reshape(
            output,
            (
                -1,
                output.shape[
                    2
                ],
            ),
        )

        x = self.fc(
            output
        )

        return (
            x,
            state,
            attention_weights,
        )

    def initialize_hidden_state(
        self,
    ):
        return tf.zeros(
            (
                self.batch_sz,
                self.dec_units,
            )
        )


# %%
tf.keras.backend.clear_session()

encoder = Encoder(
    vocab_inp_size
    + 1,
    300,
    units,
    BATCH_SIZE,
)
decoder = Decoder(
    vocab_tar_size
    + 1,
    embedding_dim,
    units,
    BATCH_SIZE,
)

# %%

optimizer = (
    tf.keras.optimizers.Adam()
)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction="none",
)


def loss_function(
    real,
    pred,
):
    mask = tf.math.logical_not(
        tf.math.equal(
            real,
            0,
        )
    )
    loss_ = loss_object(
        real,
        pred,
    )

    mask = tf.cast(
        mask,
        dtype=loss_.dtype,
    )
    loss_ *= mask

    return tf.reduce_mean(
        loss_
    )


# %%
checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(
    checkpoint_dir,
    "ckpt",
)
checkpoint = tf.train.Checkpoint(
    optimizer=optimizer,
    encoder=encoder,
    decoder=decoder,
)


# %%
@tf.function
def train_step(
    inp,
    targ,
    enc_hidden,
):
    loss = (
        0
    )

    with tf.GradientTape() as tape:
        (
            enc_output,
            enc_hidden,
        ) = encoder(
            inp,
            enc_hidden,
        )
        encoder.get_layer(
            "embedding_layer_encoder"
        ).set_weights(
            [
                embedding_matrix
            ]
        )
        dec_hidden = enc_hidden

        dec_input = tf.expand_dims(
            [
                target_lang.word_index[
                    ""
                ]
            ]
            * BATCH_SIZE,
            1,
        )

        for t in range(
            1,
            targ.shape[
                1
            ],
        ):
            (
                predictions,
                dec_hidden,
                _,
            ) = decoder(
                dec_input,
                dec_hidden,
                enc_output,
            )

            loss += loss_function(
                targ[
                    :,
                    t,
                ],
                predictions,
            )

            dec_input = tf.expand_dims(
                targ[
                    :,
                    t,
                ],
                1,
            )

    batch_loss = (
        loss
        / int(
            targ.shape[
                1
            ]
        )
    )

    variables = (
        encoder.trainable_variables
        + decoder.trainable_variables
    )

    gradients = tape.gradient(
        loss,
        variables,
    )

    optimizer.apply_gradients(
        zip(
            gradients,
            variables,
        )
    )

    return batch_loss


# %%
EPOCHS = (
    15
)

for (
    epoch
) in range(
    EPOCHS
):
    start = (
        time.time()
    )

    enc_hidden = (
        encoder.initialize_hidden_state()
    )
    total_loss = (
        0
    )

    for (
        batch,
        (
            inp,
            targ,
        ),
    ) in enumerate(
        dataset.take(
            steps_per_epoch
        )
    ):
        batch_loss = train_step(
            inp,
            targ,
            enc_hidden,
        )
        total_loss += batch_loss

        if (
            batch
            % 100
            == 0
        ):
            print(
                f"Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}"
            )
    if (
        (
            epoch
            + 1
        )
        % 2
        == 0
    ):
        checkpoint.save(
            file_prefix=checkpoint_prefix
        )

    print(
        f"Epoch {epoch+1} Loss {total_loss/steps_per_epoch:.4f}"
    )
    print(
        f"Time taken for 1 epoch {time.time()-start:.2f} sec\n"
    )

# %%

for (
    epoch
) in range(
    EPOCHS,
    20,
):
    start = (
        time.time()
    )

    enc_hidden = (
        encoder.initialize_hidden_state()
    )
    total_loss = (
        0
    )

    for (
        batch,
        (
            inp,
            targ,
        ),
    ) in enumerate(
        dataset.take(
            steps_per_epoch
        )
    ):
        batch_loss = train_step(
            inp,
            targ,
            enc_hidden,
        )
        total_loss += batch_loss

        if (
            batch
            % 100
            == 0
        ):
            print(
                f"Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}"
            )
    # saving (checkpoint) the model every 2 epochs
    if (
        (
            epoch
            + 1
        )
        % 2
        == 0
    ):
        checkpoint.save(
            file_prefix=checkpoint_prefix
        )

    print(
        f"Epoch {epoch+1} Loss {total_loss/steps_per_epoch:.4f}"
    )
    print(
        f"Time taken for 1 epoch {time.time()-start:.2f} sec\n"
    )


# %%


def evaluate(
    sentence,
):
    attention_plot = np.zeros(
        (
            max_length_targ,
            max_length_inp,
        )
    )

    sentence = preprocess(
        sentence
    )

    inputs = [
        inp_lang.word_index[
            i
        ]
        for i in sentence.split(
            " "
        )
    ]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [
            inputs
        ],
        maxlen=20,
        padding="post",
    )
    inputs = tf.convert_to_tensor(
        inputs
    )

    result = (
        ""
    )

    hidden = [
        tf.zeros(
            (
                1,
                units,
            )
        )
    ]
    (
        enc_out,
        enc_hidden,
    ) = encoder(
        inputs,
        hidden,
    )

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims(
        [
            targ_lang.word_index[
                ""
            ]
        ],
        0,
    )

    for (
        t
    ) in range(
        max_length_targ
    ):
        (
            predictions,
            dec_hidden,
            attention_weights,
        ) = decoder(
            dec_input,
            dec_hidden,
            enc_out,
        )
        # storing the attention weights to plot later on
        attention_weights = tf.reshape(
            attention_weights,
            (
                -1,
            ),
        )
        attention_plot[
            t
        ] = (
            attention_weights.numpy()
        )
        predicted_id = tf.argmax(
            predictions[
                0
            ]
        ).numpy()

        result += (
            targ_lang.index_word[
                predicted_id
            ]
            + " "
        )

        if (
            targ_lang.index_word[
                predicted_id
            ]
            == ""
        ):
            return (
                result,
                attention_plot,
            )

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims(
            [
                predicted_id
            ],
            0,
        )

    return (
        result,
        attention_plot,
    )


# %%

input_sentence = "please ensure that you use the appropriate form "
print(
    "Input sentence in english : ",
    input_sentence,
)
(
    predicted_output,
    attention_plot,
) = evaluate(
    input_sentence
)
print(
    "Predicted sentence in hindi : ",
    predicted_output,
)

# %%

input_sentence = "and do something with it to change the world "
print(
    "Input sentence in english : ",
    input_sentence,
)
(
    predicted_output,
    attention_plot,
) = evaluate(
    input_sentence
)
print(
    "Predicted sentence in hindi : ",
    predicted_output,
)
