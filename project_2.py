# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 13:00:52 2023

@author: Natalia França dos Reis & Vitor Hugo Miranda Mourao
"""

from tensorflow.keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM
from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import pickle

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
import matplotlib.pyplot as plt

def string_to_int(string, length, vocab):
    """
    Converts all strings in the vocabulary into a list of integers representing the positions of the
    input string's characters in the "vocab"

    Arguments:
    string -- input string, e.g. 'Wed 10 Jul 2007'
    length -- the number of time steps you'd like, determines if the output will be padded or cut
    vocab -- vocabulary, dictionary used to index every character of your "string"

    Returns:
    rep -- list of integers (or '<unk>') (size = length) representing the position of the string's character in the vocabulary
    """

    #make lower to standardize
    string = string.lower()
    string = string.replace(',','')

    if len(string) > length:
        string = string[:length]

    rep = list(map(lambda x: vocab.get(x, '<unk>'), string))

    if len(string) < length:
        rep += [vocab['<pad>']] * (length - len(string))

    #print (rep)
    return rep

def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):

    X, Y = zip(*dataset)

    X = np.array([string_to_int(i, Tx, human_vocab) for i in X])
    Y = [string_to_int(t, Ty, machine_vocab) for t in Y]

    Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X)))
    Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), Y)))

    return X, np.array(Y), Xoh, Yoh

def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')
        
# Define the file paths
base_path_n = "C:/Users/usuario/Desktop/DL_project/"
base_path_v = "C:/Users/vitor/Downloads/Pos grad/Doutorado/MT862 - Deep Learning/projeto 2/files/"
file_paths_n = {
    'dataset':  base_path_n + "dataset.pkl",
    'human_vocab': base_path_n + "human_vocab.pkl",
    'machine_vocab': base_path_n + "machine_vocab.pkl",
    'inv_machine_vocab': base_path_n + "inv_machine_vocab.pkl"
}
file_paths_v = {
    'dataset': base_path_v + "dataset.pkl",
    'human_vocab': base_path_v + "human_vocab.pkl",
    'machine_vocab': base_path_v + "machine_vocab.pkl",
    'inv_machine_vocab': base_path_v + "inv_machine_vocab.pkl"
}

# Load the data from the files in a single line using dictionary comprehension
data = {name: pickle.load(open(path, 'rb')) for name, path in file_paths_v.items()}

dataset = data['dataset']
human_vocab = data['human_vocab']
machine_vocab = data['machine_vocab']
inv_machine_vocab = data['inv_machine_vocab']
        
Tx = 30
Ty = 10

tf.config.list_physical_devices(device_type='GPU')

X, Y, Xoh, Yoh = preprocess_data(dataset,human_vocab,machine_vocab, Tx, Ty)

##### Attention #####
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis = -1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name = "attention_weights")
dotor = Dot(axes = 1)

def one_step_attention(a, s_prev):
    s_prev = repeator(s_prev)
    concat = concatenator([a,s_prev])
    e = densor1(concat)
    energies = densor2(e)
    alphas = activator(energies)
    context = dotor([alphas,a])
    
    return context

n_a = 32
n_s = 64

post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(len(machine_vocab), activation = softmax)

def modelf(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    X = Input(shape = (Tx, human_vocab_size))
    s0 = Input(shape = (n_s,), name = "s0")
    c0 = Input(shape = (n_s,), name = "c0")
    s = s0
    c = c0
    
    outputs = []
    a = Bidirectional(LSTM(n_a,return_sequences = True))(X)
    
    for t in range(Ty):
        context = one_step_attention(a, s)
        s, _, C = post_activation_LSTM_cell(context, initial_state = [s,c])
        out = output_layer(s)
        outputs.append(out)
    model = Model(inputs = [X,s0,c0], outputs = outputs)
    return model
        
model = modelf(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
model.summary()

opt = Adam(learning_rate = 0.005, beta_1 = 0.9, beta_2 = 0.999, weight_decay = 0.01)
model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ["accuracy"])

s0 = np.zeros((len(X), n_s))
c0 = np.zeros((len(X), n_s))

outputs = list(Yoh.swapaxes(0,1))
model.fit([Xoh, s0, c0], outputs, epochs = 50, batch_size = 100)
