import os
import pathlib
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def input_output(text):
    input_text = text[:-1]
    output_text = text[1:]
    return input_text, output_text


tf.random.set_seed(1)

sequence_length = 500  # Can cause instability at random numbers
batch_size = 1
buffer_size = 20000
embedding_vector_size = 32
lstm_units = 512
epochs = 10  # 18 min per epoch

data_dir = pathlib.Path('.')

text = ''

for book_path in data_dir.glob(os.path.join('books', 'clean', '*.txt')):
    with open(str(book_path), 'r', encoding="utf8") as fp:
        book = fp.read()
        text += book

text = re.sub(r'\n', ' ', text)
text = re.sub(r'\t', ' ', text)
text = re.sub(r'\xa0', ' ', text)
text = re.sub(r'\ufeff', ' ', text)

for i in range(100):
    text = re.sub(r'  ', ' ', text)

char_set = set(text)
num_char = len(char_set)
sorted_char_set = sorted(char_set)
int_to_char = np.array(sorted_char_set)
char_to_int = {ch: i for i, ch in enumerate(list(int_to_char))}

pickle.dump(int_to_char, open(os.path.join('objects', 'int_to_char.pkl'), 'wb'))

print('Total Length:', len(text))
print('Unique Characters:', num_char)

text_encoded = np.array([char_to_int[ch] for ch in text])
ds_text_encoded = tf.data.Dataset.from_tensor_slices(text_encoded)
ds_sequences_raw = ds_text_encoded.batch((sequence_length + 1), drop_remainder=True)
ds_sequences = ds_sequences_raw.map(input_output)
ds = ds_sequences.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

model = tf.keras.Sequential(
    [tf.keras.layers.Embedding(num_char, embedding_vector_size, batch_input_shape=[batch_size, None]),
     tf.keras.layers.LSTM(lstm_units, return_sequences=True, stateful=True),
     tf.keras.layers.Dense(num_char)])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

model.summary()

with open(os.path.join('logs', f'model_summary.txt'), 'w') as f_model_summary:
    model.summary(print_fn=(lambda x: f_model_summary.write('{}\n'.format(x))))

train_log = model.fit(ds, epochs=epochs, verbose=2)

model.save(os.path.join('objects', 'model_finish-my-sentence.h5'))

hist = train_log.history
n_epochs = np.arange(len(hist['loss'])) + 1

df_hist = pd.DataFrame.from_dict(hist)
df_hist['epoch'] = n_epochs
df_hist.to_csv(os.path.join('logs', f'train_log_history.csv'), index=False)

fig = plt.figure(figsize=(6.5, 4), dpi=600)

ax = fig.add_subplot(1, 1, 1)
ax.plot(n_epochs, hist['loss'], '-', label='Training')
ax.legend()
ax.set_xlabel('')
ax.set_ylabel('Loss')

plt.tight_layout()
plt.savefig(os.path.join('logs', f'train_log_history.png'))
