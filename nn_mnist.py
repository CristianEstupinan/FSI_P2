import gzip
import cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set


# ---------------- Visualizing some element of the MNIST dataset --------------

#import matplotlib.cm as cm
import matplotlib.pyplot as plt

#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print train_y[57]


# TODO: the neural net!!

y_data_train = one_hot(train_y[:].astype(int), 10)
y_data_valid = one_hot(valid_y[:].astype(int), 10)
y_data_test = one_hot(test_y[:].astype(int), 10)

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 10)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20

tamanoEntrenamiento = int(len(train_x))  
tamanoTest = int(len(valid_x))  

errorAnterior = 0;
contadorEstabilidad = 0
vectorErrores = []

epoch = 0

while (contadorEstabilidad < 10) or (epoch <= 20):
    for jj in xrange(tamanoEntrenamiento / batch_size):
        minimo = min(batch_size, tamanoEntrenamiento - jj * batch_size)
        batch_xs = train_x[jj * batch_size: jj * batch_size + minimo]
        batch_ys = y_data_train[jj * batch_size: jj * batch_size + minimo]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    batch_xv = valid_x
    batch_yv = y_data_valid
    error = sess.run(loss, feed_dict={x: batch_xv, y_: batch_yv})
    vectorErrores.append(error)

    if (epoch == 0):
        errorAnterior = error
    elif (error >= errorAnterior * 0.95):
        contadorEstabilidad += 1
    else:
        contadorEstabilidad = 0
        errorAnterior = error

    print "Estabilizacion: %d" % contadorEstabilidad


    print "Epoch #:", epoch, "Error: ", error
    print "----------------------------------------------------------------------------------"
    epoch += 1

batch_xt = test_x
batch_yt = y_data_test

contadorFallos = 0
result = sess.run(y, feed_dict={x: batch_xt})
for b, r in zip(batch_yt, result):
    if np.argmax(b) != np.argmax(r):
        contadorFallos += 1


print "El numero de fallos obtenido es de: %d" % contadorFallos
resultado = contadorFallos/float(tamanoTest)
print "El porcentaje es de: %f" % resultado

plt.figure()
vectorErrores = np.array(vectorErrores)
plt.plot(vectorErrores)
plt.show()