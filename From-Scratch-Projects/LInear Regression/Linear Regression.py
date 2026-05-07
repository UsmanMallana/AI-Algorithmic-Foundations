import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100).astype(np.float32)
Y = 2*X+1+np.random.normal(0,0.1,X.shape)

class LinearModel(tf.Module):
    def __init__(self):
        self.w = tf.Variable(tf.random.normal([1],dtype=tf.float32))
        self.b = tf.Variable(tf.zeros([1],dtype=tf.float32))
    def __call__(self,x):
        return self.w*x+self.b

def calculate_loss(y_true,y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred))

model = LinearModel()

optimizer = tf.optimizers.SGD(learning_rate=0.01)

epochs = 1000

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = calculate_loss(Y,y_pred)
    gradients = tape.gradient(loss,[model.w,model.b])
    optimizer.apply_gradients(zip(gradients,[model.w,model.b]))
    if epoch%10 == 0:
        print(f"The loss is {loss} at epoch {epoch}")

plt.scatter(X,Y,label="Data")
plt.plot(X,y_pred,color='red',label='Fitted Line')
plt.legend()
plt.show()