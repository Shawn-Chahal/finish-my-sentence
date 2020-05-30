import copy
import os
import pickle
import numpy as np
import tensorflow as tf


def text_generator(text, scale_factor):
    len_generated_text = 100

    if (text[-1] != ' ') and (len(text) > 2):
        text = text + ' '

    if len(text) % ngram != 0:
        text = ' ' + text

    text_token = []

    for i in range(0, len(text), ngram):
        if text[i: i + ngram] in char_to_int:
            text_token.append(text[i: i + ngram])

    if len(text_token) == 0:
        text_token.append('Th')

    encoded_input = [char_to_int[s] for s in text_token]
    encoded_input = np.reshape(encoded_input, (1, -1))

    y = np.zeros((batch_size, encoded_input.shape[1]))
    y[0, :] = encoded_input

    z = np.zeros((batch_size, 1))
    max_input_length = 1000  # Based on speed, reduce if compute time is too long
    encoded_input = y[:, -max_input_length:]
    output_text = copy.copy(text_token)

    model.reset_states()
    for i in range(len_generated_text):
        logits = model(encoded_input)[0, -1]

        scaled_logits = logits * scale_factor
        new_char_indx = tf.random.categorical([scaled_logits], num_samples=1)
        output_text += str(int_to_char[new_char_indx[0, 0]])

        if encoded_input.shape != (batch_size, 1):
            encoded_input = np.zeros((batch_size, 1))
        encoded_input[0, 0] = new_char_indx

    output_text = ''.join(output_text)

    if output_text[0] == ' ':
        output_text = output_text[1:]

    output_text += '...'

    return output_text


model_dir = 'lite'

tf.random.set_seed(1)

model = tf.keras.models.load_model(os.path.join(model_dir, 'objects', 'model.h5'))
char_to_int = pickle.load(open(os.path.join(model_dir, 'objects', 'char_to_int.pkl'), 'rb'))
int_to_char = pickle.load(open(os.path.join(model_dir, 'objects', 'int_to_char.pkl'), 'rb'))
batch_size = pickle.load(open(os.path.join(model_dir, 'objects', 'batch_size.pkl'), 'rb'))
ngram = pickle.load(open(os.path.join(model_dir, 'objects', 'ngram.pkl'), 'rb'))

user_message = ["It was the last night of their journey to",
                "He set off for",
                "The moon",
                "She",
                "T"
                ]

scales = np.linspace(1.0, 2.0, 5)

for scale_factor in scales:
    print(f'---Scale Factor: {scale_factor}---')
    print('--------------------')
    for message in user_message:
        print(text_generator(text=message, scale_factor=scale_factor))
        print('--------------------')
