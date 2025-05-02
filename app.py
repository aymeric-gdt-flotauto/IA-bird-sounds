from models.config import models
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt


# config var to models archi
epochs=30
target_size=640
batch_size=32
train_dir=None
seed=42

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 #utilisation de 20% des données 
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(target_size, target_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=seed
)
val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(target_size, target_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',  
        seed=seed
)


# verifications
print("Classes d'entraînement:", train_gen.class_indices)
print("Classes de validation:", val_gen.class_indices)
print("Nombre de classes d'entraînement:", len(train_gen.class_indices))
print("Nombre de classes de validation:", len(val_gen.class_indices))

num_classes = len(train_gen.class_indices)

# callback to stop early
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)
inputs = tf.layers.Input(shape=(640, 640, 3))
model = models(num_classes,inputs)

history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=[early_stopping]
)

# Visualisation des résultats
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Précision du modèle')
plt.ylabel('Précision')
plt.xlabel('Époque')
plt.legend(['Entraînement', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Perte du modèle')
plt.ylabel('Perte')
plt.xlabel('Époque')
plt.legend(['Entraînement', 'Validation'], loc='upper right')

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

model.save('model_01.keras')