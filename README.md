# Machine_transliteration
Machine Transliteration ENGLISH -> ASSAMESE 

import pandas as pd
import numpy as np
import string
from string import digits
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import re
import os
import io
import time
import fasttext
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import json


# Load the Pretrained English FastText model
model_en = fasttext.load_model("C:\\Users\\Sk Shamim Aktar\\Desktop\extracted\\cc.en.300.bin")


#showing dimension
model_en.get_dimension()



# Read the training data from JSON file
train_data = pd.read_json("C:\\Users\\Sk Shamim Aktar\\Desktop\extracted\\asm_train.json", lines=True)


# Extract English and Assamese words from the train_data
ass_words = train_data['native word']
eng_words = train_data['english word']


# Prepare the training data file
with open("C:\\Users\\Sk Shamim Aktar\\Desktop\extracted.txt", 'w', encoding='utf-8') as f:
    for text in ass_words:
        f.write(text + '\n')


# Train the FastText model on the training data and save the model
ass_model = fasttext.train_unsupervised("C:\\Users\\Sk Shamim Aktar\\Desktop\extracted.txt", minn=1, maxn=5, dim=100)
ass_model.save_model("C:\\Users\\Sk Shamim Aktar\\Desktop\extracted.bin")


ass_model = fasttext.load_model("C:\\Users\\Sk Shamim Aktar\\Desktop\extracted.bin")


#This line calls the get_word_vector() method of the "ass_model",
        #The method returns the word vector representation of the given word.
ass_model.get_word_vector("ভাল")


# Function to split a word into character
def split_word_into_characters(word):
    return ' '.join(list(word))


# Apply character splitting to English dataframe
eng_words = eng_words.apply(split_word_into_characters)

# Apply character splitting to Assamese dataframe
ass_words = ass_words.apply(split_word_into_characters)


# Tokenization(It uses a character-level approach, which means it tokenizes the text at the character level
                        #For example, the word "hello" would be tokenized into ['h', 'e', 'l', 'l', 'o']. )
#Tokenize the English words:
eng_tokenizer = Tokenizer(char_level=True)
eng_tokenizer.fit_on_texts(eng_words)
eng_sequences = eng_tokenizer.texts_to_sequences(eng_words)


#Tokenize the Assamese words:
ass_tokenizer = Tokenizer(char_level=True)
ass_tokenizer.fit_on_texts(ass_words)
ass_sequences = ass_tokenizer.texts_to_sequences(ass_words)


# Padding (to ensure sequences have the same length)
max_seq_length = max(max(len(seq) for seq in eng_sequences), max(len(seq) for seq in ass_sequences))
eng_sequences_padded = pad_sequences(eng_sequences, maxlen=max_seq_length, padding='post')
ass_sequences_padded = pad_sequences(ass_sequences, maxlen=max_seq_length, padding='post')


# Create word embeddings using FastText model
embedding_dimension = 300

eng_word_embeddings = np.zeros((len(eng_tokenizer.word_index) + 1, embedding_dimension))
ass_word_embeddings = np.zeros((len(ass_tokenizer.word_index) + 1, embedding_dimension))

for word, index in eng_tokenizer.word_index.items():
    if word in model_en:
        eng_word_embeddings[index] = model_en.get_word_vector(word)

for word, index in ass_tokenizer.word_index.items():
    if word in ass_model:
        ass_word_embeddings[index] = ass_model.get_word_vector(word)


eng_word_embeddings.shape


ass_word_embeddings.shape

from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
# Define Encoder-Decoder Model

#Encode
encoder_input = Input(shape=(max_seq_length,))
encoder_embedding = Embedding(len(eng_tokenizer.word_index) + 1, embedding_dimension, weights=[eng_word_embeddings], trainable=False)(encoder_input)
encoder = LSTM(256)(encoder_embedding)

# Decoder
decoder_input = Input(shape=(max_seq_length,))
decoder_embedding = Embedding(len(ass_tokenizer.word_index) + 1, embedding_dimension, weights=[ass_word_embeddings], trainable=False)(decoder_input)
decoder = LSTM(256, return_sequences=True)(decoder_embedding, initial_state=[encoder, encoder])

# Output
decoder_output = Dense(len(ass_tokenizer.word_index) + 1, activation='softmax')(decoder)
# Define the model
model = Model([encoder_input, decoder_input], decoder_output)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.summary()

#train the model
model.fit([eng_sequences_padded, ass_sequences_padded], ass_sequences_padded, batch_size=64, epochs=50, validation_split=0.2)


#Read the test data from a JSON file;
test_data_frame = pd.read_json("C:\\Users\\Sk Shamim Aktar\\Desktop\extracted\\asm_test.json", lines=True)


#Extract the Assamese and English words from the test data:
test_ass_words = test_data_frame['native word']
test_eng_words = test_data_frame['english word']


#Apply character splitting to the English and Assamese test words:
test_eng_words = test_eng_words.apply(split_word_into_characters)
test_ass_words = test_ass_words.apply(split_word_into_characters)


#Tokenize and pad the test sequences:
test_eng_sequences = eng_tokenizer.texts_to_sequences(test_eng_words)
test_ass_sequences = ass_tokenizer.texts_to_sequences(test_ass_words)

test_eng_sequences_padded = pad_sequences(test_eng_sequences, maxlen=max_seq_length, padding='post')
test_ass_sequences_padded = pad_sequences(test_ass_sequences, maxlen=max_seq_length, padding='post')


#Evaluate the model:
evaluation = model.evaluate([test_eng_sequences_padded, test_ass_sequences_padded], test_ass_sequences_padded)

for metric_name, metric_value in zip(model.metrics_names, evaluation):
    print(metric_name + ':', metric_value)


#Define a function to predict transliteration:
def predict_transliteration(input_word):
    input_word = split_word_into_characters(input_word)
    input_sequence = eng_tokenizer.texts_to_sequences([input_word])
    input_sequence_padded = pad_sequences(input_sequence, maxlen=max_seq_length, padding='post')
    predicted_sequence = model.predict([input_sequence_padded, input_sequence_padded])
    predicted_word = ''
    for seq in predicted_sequence[0]:
        index = np.argmax(seq)
        if index in ass_tokenizer.index_word:
            predicted_char = ass_tokenizer.index_word[index]
            if predicted_char != ' ':
                predicted_word += predicted_char
        else:
            predicted_word += ''
    return predicted_word


#Prompt the user to enter an English word and predict its transliteration;
input_word = input("Enter an English word: ")
predicted_word = predict_transliteration(input_word)
print('Input:', input_word)
print('Predicted Transliteration:', predicted_word)


