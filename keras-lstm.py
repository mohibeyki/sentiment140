from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential

from sentiment140 import parse_data, generator

total_samples = 10000
max_features = 4403  # 11135
max_len = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 64
total_batches_per_epoch = total_samples // batch_size

# x_train, y_train = parse_data('sample_set.csv')
x_test, y_test = parse_data('test_set.csv')

print('Building model')
model = Sequential()
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(batch_size, max_features)))
model.add(Dense(3, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint('weights.{epoch:02d}.hdf5', monitor='val_loss', verbose=1)

model.fit_generator(
    generator=generator('sample_set.csv', batch_size),
    steps_per_epoch=total_batches_per_epoch,
    epochs=10,
    verbose=False,
    callbacks=[checkpoint],
    validation_data=generator('test_set.csv', batch_size),
    validation_steps=498 // batch_size
)

# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=10,
#           validation_data=(x_test, y_test))

score, acc = model.evaluate(
    x_test,
    y_test,
    batch_size=batch_size
)

print('Test score:', score)
print('Test accuracy:', acc)
