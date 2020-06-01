import os
import pickle
import numpy as np
import tensorflow as tf


def text_generator(text, scale_factor):
    len_generated_text = int(150 / ngram)

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

    max_input_length = 200  # Lower this if running into performance issues
    encoded_input = encoded_input[:, -max_input_length:]

    generated_text = ''

    model.reset_states()
    for i in range(len_generated_text):
        logits = model(encoded_input)[0, -1]

        scaled_logits = logits * scale_factor
        new_char_indx = tf.random.categorical([scaled_logits], num_samples=1)
        generated_text += str(int_to_char[new_char_indx[0, 0]])

        encoded_input = [[new_char_indx]]

    generated_text = ''.join(generated_text)

    if text[0] == ' ':
        text = text[1:]

    generated_text += '...'

    return [text, generated_text]


tf.random.set_seed(1)

model = tf.keras.models.load_model(os.path.join('objects', 'model.h5'))
char_to_int = pickle.load(open(os.path.join('objects', 'char_to_int.pkl'), 'rb'))
int_to_char = pickle.load(open(os.path.join('objects', 'int_to_char.pkl'), 'rb'))
ngram = pickle.load(open(os.path.join('objects', 'ngram.pkl'), 'rb'))

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
        print(''.join(text_generator(text=message, scale_factor=scale_factor)))
        print('--------------------')
