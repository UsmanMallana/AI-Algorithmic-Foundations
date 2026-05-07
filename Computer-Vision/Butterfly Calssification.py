import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers,models
import os

save_path = 'Tensorflow/Dataset'
butterfly_dataset = tf.data.Dataset.load(save_path)

model = models.Sequential([
    layers.Conv2D(64,(3,3),activation='relu',input_shape=(128,128,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dense(75,activation='softmax')
])

model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

history = model.fit(butterfly_dataset,epochs=2)

model.save('Tensorflow\Butterfly Classification Model.h5')

loss, accuracy = model.evaluate(butterfly_dataset)

print(f"The model's accuracy is {accuracy} and loss is {loss}")