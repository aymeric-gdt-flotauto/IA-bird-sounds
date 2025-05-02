from models.config import models
import tensorflow as tf






# callback to stop early
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)
inputs = tf.layers.Input(shape=(96, 96, 3))
model = models(inputs)

history = model.fit(
    
)

