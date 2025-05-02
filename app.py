from models.config import models
import tensorflow as tf

epochs=30





# callback to stop early
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)
inputs = tf.layers.Input(shape=(640, 640, 3))
model = models(inputs)

history = model.fit(
    epochs=epochs,
    callbacks=[early_stopping]
)

