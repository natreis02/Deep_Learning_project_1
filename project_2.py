# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 13:00:52 2023

@author: Natalia França dos Reis & Vitor Hugo Miranda Mourao
"""

from tensorflow.keras.layers import Bidirectional, Concatenate, Dot, Input, LSTM
from tensorflow.keras.layers import RepeatVector, Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.activations import softmax
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

    return rep

def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):

    X, Y = zip(*dataset)

    X = np.array([string_to_int(i, Tx, human_vocab) for i in X])
    Y = [string_to_int(t, Ty, machine_vocab) for t in Y]

    Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X)))
    Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), Y)))

    return X, np.array(Y), Xoh, Yoh

# def softmax(x, axis=1):
#     """Softmax activation function.
#     # Arguments
#         x : Tensor.
#         axis: Integer, axis along which the softmax normalization is applied.
#     # Returns
#         Tensor, output of softmax transformation.
#     # Raises
#         ValueError: In case `dim(x) == 1`.
#     """
#     ndim = K.ndim(x)
#     if ndim == 2:
#         return K.softmax(x)
#     elif ndim > 2:
#         e = K.exp(x - K.max(x, axis=axis, keepdims=True))
#         s = K.sum(e, axis=axis, keepdims=True)
#         return e / s
#     else:
#         raise ValueError('Cannot apply softmax to a tensor that is 1D')
        
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
activator = Activation('softmax', name = "attention_weights")
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
# model.summary()

opt = Adam(learning_rate = 0.005, beta_1 = 0.9, beta_2 = 0.999, weight_decay = 0.01)
model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ["accuracy"])

s0 = np.zeros((len(X), n_s))
c0 = np.zeros((len(X), n_s))

outputs = list(Yoh.swapaxes(0,1))
model.fit([Xoh, s0, c0], outputs, epochs = 25, batch_size = 100)

##### Model Prediction ######

fake = Faker()
Faker.seed(1992)
random.seed(1996)

# Define format of the data we would like to generate
FORMATS = ['short',
           'medium',
           'long',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'd MMM YYY', 
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY']

# change this if you want it to work with another language
LOCALES = ['en_US']

def load_date():
    """
        Loads some fake dates 
        :returns: tuple containing human readable string, machine readable string, and date object
    """
    dt = fake.date_object()

    try:
        human_readable = format_date(dt, format=random.choice(FORMATS),  locale='en_US') # locale=random.choice(LOCALES))
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',','')
        machine_readable = dt.isoformat()

    except AttributeError as e:
        return None, None, None

    return human_readable, machine_readable, dt

def load_dataset(m):
    """
        Loads a dataset with m examples and vocabularies
        :m: the number of examples to generate
    """

    human_vocab = set()
    machine_vocab = set()
    dataset = []


    for i in tqdm(range(m)):
        h, m, _ = load_date()
        if h is not None:
            dataset.append((h, m))
            human_vocab.update(tuple(h))
            machine_vocab.update(tuple(m))

    return dataset

m = 1000
dataset_test = load_dataset(m)

EXAMPLES = [example[0] for example in dataset_test]
TARGETS = [example[1] for example in dataset_test] 
s00 = np.zeros((1, n_s))
c00 = np.zeros((1, n_s))

correct_predictions = 0

for i, example in enumerate(EXAMPLES):
    print(i,"\n")
    # Convert the raw date string into a one-hot encoded version
    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))
    source = np.expand_dims(source, axis=0)

    # Predict the output using the model
    prediction = model.predict([source, s00, c00])
    prediction = np.argmax(prediction, axis=-1)

    # Iterate over the Ty dimension to get predictions
    output = [inv_machine_vocab[int(i)] for i in prediction]
    predicted_date = ''.join(output)
    
    # Compare the predicted date to the actual date
    actual_date = TARGETS[i]
    if predicted_date == actual_date:
        correct_predictions += 1
    else:
        print("source:", example)
        print("output:", predicted_date)
        print("actual:", actual_date, "\n")

# Calculate the accuracy
accuracy = correct_predictions / len(EXAMPLES)
print("Accuracy:", accuracy)

##### PLOT #####

def int_to_string(ints, inv_vocab):
    """
    Output a machine readable list of characters based on a list of indexes in the machine's vocabulary

    Arguments:
    ints -- list of integers representing indexes in the machine's vocabulary
    inv_vocab -- dictionary mapping machine readable indexes to machine readable characters 

    Returns:
    l -- list of characters corresponding to the indexes of ints thanks to the inv_vocab mapping
    """

    l = [inv_vocab[i] for i in ints]
    return l

def plot_attention_map(model, input_vocabulary, inv_output_vocabulary, text, n_s = 64, Tx = 30, Ty = 10):
    """
    Plot the attention map.

    """
    attention_map = np.zeros((Ty, Tx))

    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    layer = model.layers[7]

    encoded = np.array(string_to_int(text, Tx, input_vocabulary)).reshape((1, Tx))
    encoded = np.array(list(map(lambda x: to_categorical(x, num_classes=len(input_vocabulary)), encoded)))

    f = K.function(model.inputs, [layer.get_output_at(t) for t in range(Ty)])
    r = f([encoded, s0, c0])

    for t in range(Ty):
        for t_prime in range(Tx):
            attention_map[t][t_prime] = r[t][0,t_prime,0]

    # Normalize attention map
    row_max = attention_map.max(axis=1)
    attention_map = attention_map / row_max[:, None]

    prediction = model.predict([encoded, s0, c0])

    predicted_text = []
    for i in range(len(prediction)):
        predicted_text.append(int(np.argmax(prediction[i], axis=1)))

    predicted_text = list(predicted_text)
    predicted_text = int_to_string(predicted_text, inv_output_vocabulary)
    text_ = list(text)

    # get the lengths of the string
    input_length = len(text)
    output_length = Ty

    # Plot the attention_map
    plt.clf()
    f = plt.figure(figsize=(8, 8.5))
    ax = f.add_subplot(1, 1, 1)

    # add image
    i = ax.imshow(attention_map, interpolation='nearest', cmap='Blues')

    # add colorbar
    cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
    cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
    cbar.ax.set_xlabel('Alpha value (Probability output of the "softmax")', labelpad=2)

    # add labels
    ax.set_yticks(range(output_length))
    ax.set_yticklabels(predicted_text[:output_length])

    ax.set_xticks(range(input_length))
    ax.set_xticklabels(text_[:input_length], rotation=45)

    ax.set_xlabel('Input Sequence')
    ax.set_ylabel('Output Sequence')

    # add grid and legend
    ax.grid()

    # Save the figure instead of showing it
    f.savefig('attention_map.png', bbox_inches='tight')

    return attention_map

text = "15 11 2005"
# Call the function with the correct parameters
plot_attention_map(model, human_vocab, inv_machine_vocab, text, n_s, Tx, Ty)
