import os
import pickle
import numpy as np
import tensorflow as tf


def text_generator(text, scale_factor):
    if text[-1] != ' ':
        text = text + ' '

    text_token = ""

    for i in range(0, len(text)):
        if text[i] in char_to_int:
            text_token += text[i]

    if len(text_token) == 0:
        text_token += 'Th'

    encoded_input = [char_to_int[s] for s in text_token]
    encoded_input = np.reshape(encoded_input, (1, -1))
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

len_generated_text = 200
max_input_length = 200  # Lower this if running into performance issues

model = tf.keras.models.load_model(os.path.join('objects', 'model_finish-my-sentence.h5'))
int_to_char = pickle.load(open(os.path.join('objects', 'int_to_char.pkl'), 'rb'))
char_to_int = {ch: i for i, ch in enumerate(list(int_to_char))}

user_message = ["It was the last night of their journey to",
                "He set off for",
                "The moon",
                "She",
                "T"
                ]

scales = np.linspace(1, 3, 5)

for scale_factor in scales:
    print(f'---Scale Factor: {scale_factor}---')
    print('--------------------')
    for message in user_message:
        print(''.join(text_generator(message, scale_factor)))
        print('--------------------')
