
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2

dataset = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = dataset.load_data()
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
recmodel = tf.keras.models.Sequential()
recmodel.add(tf.keras.layers.Flatten(input_shape=(28,28)))
recmodel.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
recmodel.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
recmodel.add(tf.keras.layers.Dense(units=128, activation=tf.nn.softmax))
recmodel.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
recmodel.fit(x_train, y_train, epochs = 3)
modaccuracy, modloss = recmodel.evaluate(x_test, y_test)
print(modaccuracy)
print(modloss)
recmodel.save('digitrec.recmodel')


'''
for i in range(2,4):
    image = cv2.imread(f'{i}.png')[:,:,0]
    image = np.invert(np.array([image]))
    predic = recmodel.predict(image)
    print(f'The result is: {np.argmax(predic)}')
    plt.imshow(image[0], cmap=plt.cm.binary)
    plt.show()
'''
