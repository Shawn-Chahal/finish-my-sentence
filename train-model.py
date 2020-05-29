import copy
import os
import pathlib
import pickle
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


def input_output(text):
    input_text = text[:-1]
    output_text = text[1:]
    return input_text, output_text


tuning_hyperparameters = False
initial_training = True

tf.random.set_seed(1)

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

ngram = 2

if ngram > 1:

    text_token = []
    for i in range(0, len(text), ngram):
        text_token.append(text[i: i + ngram])
    text = copy.copy(text_token)

char_set = set(text)
num_char = len(char_set)
sorted_char_set = sorted(char_set)

char_to_int = {ch: i for i, ch in enumerate(sorted_char_set)}

print('Total Length:', len(text))
print('Unique Characters:', num_char)

int_to_char = np.array(sorted_char_set)

text_encoded = np.array([char_to_int[ch] for ch in text])
ds_text_encoded = tf.data.Dataset.from_tensor_slices(text_encoded)

sequence_length = 500

ds_sequences_raw = ds_text_encoded.batch((sequence_length + 1), drop_remainder=True)
ds_sequences = ds_sequences_raw.map(input_output)

batch_size = 1
buffer_size = 10000
epochs = 1

pickle.dump(char_to_int, open(os.path.join('objects', 'char_to_int.pkl'), 'wb'))
pickle.dump(int_to_char, open(os.path.join('objects', 'int_to_char.pkl'), 'wb'))
pickle.dump(batch_size, open(os.path.join('objects', 'batch_size.pkl'), 'wb'))
pickle.dump(ngram, open(os.path.join('objects', 'ngram.pkl'), 'wb'))

ds = ds_sequences.batch(batch_size, drop_remainder=True)

num_batch = int((len(text) / (sequence_length + 1)) / batch_size)

if tuning_hyperparameters:
    num_test = int(0.2 * num_batch)
    num_valid = int(0.1 * num_batch)
else:
    num_test = int(0.05 * num_batch)
    num_valid = int(0.05 * num_batch)

ds_test = ds.take(num_test)
ds_train_valid = ds.skip(num_test)
ds_valid = ds_train_valid.take(num_valid)
ds_train = ds_train_valid.skip(num_valid)

if initial_training:
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(num_char, 256, batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(768, return_sequences=True, stateful=True),
        tf.keras.layers.Dense(num_char)])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


else:
    model = tf.keras.models.load_model(os.path.join('objects', 'model.h5'))

model.summary()

with open(os.path.join('logs', f'model_summary.txt'), 'w') as f_model_summary:
    model.summary(print_fn=(lambda x: f_model_summary.write('{}\n'.format(x))))

train_log = model.fit(ds_train, epochs=epochs, verbose=1, validation_data=ds_valid)

test_log = model.evaluate(ds_test, verbose=0)

with open(os.path.join('logs', f'test_log.txt'), 'w') as f_test_log:
    f_test_log.write('Test Loss:     {:.3f}\nTest Accuracy: {:.3f}'.format(test_log[0], test_log[1]))

model.save(os.path.join('objects', 'model.h5'))

hist = train_log.history

n_epochs = np.arange(len(hist['loss'])) + 1

df_hist = pd.DataFrame.from_dict(hist)
df_hist['epoch'] = n_epochs
df_hist.to_csv(os.path.join('logs', f'train_log_history.csv'), index=False)

fig = plt.figure(figsize=(6.5, 6.5), dpi=600)

ax = fig.add_subplot(2, 1, 1)
ax.plot(n_epochs, hist['loss'], '-', label='Training')
ax.plot(n_epochs, hist['val_loss'], '--', label='Validation')
ax.legend()
ax.set_xlabel('')
ax.set_ylabel('Loss')

ax = fig.add_subplot(2, 1, 2)
ax.plot(n_epochs, hist['accuracy'], '-', label='Training')
ax.plot(n_epochs, hist['val_accuracy'], '--', label='Validation')
ax.legend()
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')

plt.tight_layout()
plt.savefig(os.path.join('logs', f'train_log_history.png'))
