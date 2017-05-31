import keras
from keras.preprocessing import sequence

from sentiment140 import parse_data

# total_samples = 1600000
max_features = 40000  # 4403  # 11135
max_len = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 128
# total_batches_per_epoch = total_samples // batch_size

print('Loading train data')
x_train, y_train = parse_data(max_features, 'train_set_shuffled.csv')
print('Loading test data')
x_test, y_test = parse_data(max_features, 'test_set.csv')

print('Pad sequencing')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Building model')
model = keras.models.Sequential()
model.add(keras.layers.embeddings.Embedding(max_features, 256))
model.add(keras.layers.LSTM(256, dropout=0.2, recurrent_dropout=0.2))
model.add(keras.layers.Dense(3, activation='softmax'))

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)

checkpoint = keras.callbacks.ModelCheckpoint('weights.{epoch:02d}.hdf5', monitor='val_loss', verbose=1)

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
    epochs=15,
    verbose=True,
    callbacks=[checkpoint],
)

score, acc = model.evaluate(
    x_test,
    y_test,
    batch_size=batch_size,
    verbose=True
)

print('Test score:', score)
print('Test accuracy:', acc)
