import tensorflow as tf
from tensorflow.keras import datasets,layers,models
import matplotlib.pyplot as plt

(train_images,train_labels),(test_images,test_labels) = datasets.mnist.load_data()
train_images = (train_images/255.0).astype("float32")
test_images = (test_images/255.0).astype("float32")

model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='tanh',input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='tanh'),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])

model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

history = model.fit(train_images,train_labels,epochs=2,validation_data=(test_images,test_labels))

loss,accuracy = model.evaluate(test_images,test_labels)

model.save("Tensorflow\DigitRecognition.keras")

print(f"The loss of the model is {loss} and the accuracy is {accuracy}")

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()