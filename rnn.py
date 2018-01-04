import heapq
import random
import numpy as np
import read_files

from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard

NVOCAB = 10000

random.seed(500)
np.random.seed(500)

NEPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.01

MODEL_FILE = 'models' + "/models-rnn.h5"

x_train, y_train, x_test, y_test = read_files.split_data(n=10, ntest=10000,
                                                         train_amount=0.01)

word_vectors, token_with_index_dict, indx_to_word = read_files.get_processed_data()

embedding_dim = len(word_vectors['a'])

embedding_matrix = np.zeros((NVOCAB + 1, embedding_dim))

for word, value in token_with_index_dict.items():
    if value > NVOCAB:
        continue
    word_vector_value = word_vectors.get(word)
    if word_vector_value is not None:
        embedding_matrix[value] = word_vector_value

del word_vectors
del read_files.word_vectors

model = Sequential()

embedding_layer = Embedding(input_dim=NVOCAB + 1, output_dim=100,
                            input_length=10 - 1, weights=[embedding_matrix])

model.add(embedding_layer)

model.layers[0].trainable = True

model.add(GRU(100, return_sequences=True, kernel_initializer="uniform"))
model.add(Dropout(0.1))
model.add(GRU(100, return_sequences=True, kernel_initializer="uniform"))
model.add(Dropout(0.1))
model.add(GRU(100, kernel_initializer="uniform"))
model.add(Dropout(0.1))

model.add(Dense(NVOCAB))

model.add(Activation('softmax'))

try:
    print "Trying to load from checkpoint"
    model.load_weights(MODEL_FILE)
    print "Success"
except:
    print "Failed to load from checkpoint"

metrics = ['accuracy']

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=metrics)


def get_best_word_probabilites(probs, num_word_to_predict):
    index_probabilities = [(indx, prob) for indx, prob in enumerate(probs[0])]

    best_pairs = heapq.nlargest(num_word_to_predict, index_probabilities,
                                key=lambda pair: pair[1])

    total = sum([prob for _, prob in best_pairs])

    best_word_probs = [(indx, prob / total) if total != 0
                       else (indx, 0.0) for indx, prob in best_pairs]

    return best_word_probs


def choose_words(index_probabilities, num_word_to_predict):
    all_index, all_probs = map(list, zip(*index_probabilities))

    list_of_words = np.random.choice(all_index, num_word_to_predict, all_probs)

    return list_of_words


def generate_text(model, data, n, nwords=20, num_word_to_predict=3):
    x = np.zeros((1, n - 1), dtype=int)

    word_positon = random.randint(1, NVOCAB)

    words = []
    for _ in range(nwords):
        x = np.roll(x, -1)

        x[0, -1] = word_positon

        list_of_probabilities = model.predict_proba(x, verbose=0)

        word_probability_matrix = get_best_word_probabilites(
            list_of_probabilities,
            num_word_to_predict)

        index_of_words = choose_words(word_probability_matrix, 1)[0]

        words.append(data[index_of_words])

    sentence = ' '.join(words)
    return sentence


tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1,
                          write_graph=True, write_images=False,
                          embeddings_freq=1)

checkpoint = ModelCheckpoint(MODEL_FILE, monitor='val_acc', save_best_only=True,
                             mode='max')

early_stopping = EarlyStopping(monitor='val_acc', patience=10)

callbacks = [checkpoint, early_stopping, tensorboard]

history = None

print('Starting Training')

try:
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE,
                        epochs=NEPOCHS,
                        validation_split=VALIDATION_SPLIT,
                        callbacks=callbacks)
except KeyboardInterrupt:
    pass

print('Final epoch generated text:', generate_text(model, indx_to_word, 10))

print('Model History Infromation')

print(history.history)

print('Generate Final Text:')

for i in range(10):
    print(generate_text(model, indx_to_word, 10, 20, 3))

loss, accuracy = model.evaluate(x_test, y_test, BATCH_SIZE, verbose=0)

print("Test loss:", loss)

print("Test accuracy:", accuracy)
