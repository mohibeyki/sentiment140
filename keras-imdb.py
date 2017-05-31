from keras.callbacks import ModelCheckpoint
from keras.datasets import imdb
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence

max_features = 20000
max_len = 80
batch_size = 128

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(
    'weights.{epoch:02d}.hdf5',
    monitor='val_loss',
    verbose=1
)

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=15,
    verbose=True
)

score, acc = model.evaluate(
    x_test,
    y_test,
    batch_size=batch_size
)

print('Test score:', score)
print('Test accuracy:', acc)
