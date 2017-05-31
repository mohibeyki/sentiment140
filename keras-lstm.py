from keras.callbacks import ModelCheckpoint
from keras.datasets import imdb
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence

# total_samples = 1600000
max_features = 20000  # 4403  # 11135
max_len = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 128
# total_batches_per_epoch = total_samples // batch_size

# x_train, y_train = parse_data('sample_set.csv')
# x_test, y_test = parse_data('test_set.csv')

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Building model')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint('weights.{epoch:02d}.hdf5', monitor='val_loss', verbose=1)

# model.fit_generator(
#     generator=generator('sample_set.csv', batch_size),
#     steps_per_epoch=total_batches_per_epoch,
#     epochs=10,
#     verbose=False,
#     callbacks=[checkpoint],
#     validation_data=generator('test_set.csv', batch_size),
#     validation_steps=498 // batch_size
# )

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=15
)

score, acc = model.evaluate(
    x_test,
    y_test,
    batch_size=batch_size
)

print('Test score:', score)
print('Test accuracy:', acc)
