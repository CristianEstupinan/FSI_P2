import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

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
tamanoEntrenamiento = int(len(x_data)*0.7)
tamanoTest = int(len(x_data)*0.15)
tamanoValidacion = len(x_data) - tamanoEntrenamiento - tamanoTest

errorAnterior = 0
contadorEstabilizacion = 0
vectorErrores = []

epoch = 0

while (contadorEstabilizacion < 15) or (epoch <= 30):
    for jj in xrange(int(tamanoEntrenamiento/batch_size)):
        minimo = min(batch_size, tamanoEntrenamiento - jj * batch_size)
        batch_xs = x_data[jj * batch_size: jj * batch_size + minimo]
        batch_ys = y_data[jj * batch_size: jj * batch_size + minimo]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    batch_xv = x_data[jj * batch_size + minimo: jj * batch_size + minimo + tamanoValidacion]
    batch_yv = y_data[jj * batch_size + minimo: jj * batch_size + minimo + tamanoValidacion]
    error = sess.run(loss, feed_dict={x: batch_xv, y_: batch_yv})
    vectorErrores.append(error)

    #Comprobacion de la estabilizacion

    if (epoch == 0):
        errorAnterior = error
    elif (error >= errorAnterior * 0.95):
        contadorEstabilizacion += 1
    else:
        contadorEstabilizacion = 0
        errorAnterior = error

    print "Estabilizacion: %d" % contadorEstabilizacion


    print "Epoch #:", epoch, "Error: ", error
    result = sess.run(y, feed_dict={x: batch_xv})
    for b, r in zip(batch_yv, result):
        print b, "-->", r
    print "----------------------------------------------------------------------------------"
    epoch += 1

batch_xt = x_data[len(x_data) - tamanoTest:]
batch_yt = y_data[len(y_data) - tamanoTest:]

contadorFallos = 0
result = sess.run(y, feed_dict={x: batch_xt})
for b, r in zip(batch_yt, result):
    if np.argmax(b) != np.argmax(r):
        contadorFallos += 1


print "El numero de fallos obtenido es de: %d" % contadorFallos
print "El porcentaje es de: %f" % float(contadorFallos/tamanoTest)

plt.figure()
vectorErrores = np.array(vectorErrores)
plt.plot(vectorErrores)
plt.show()

