from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
import numpy as np
import tensorflow as tf
import sys


class NeuralNetwork:

# Neural Network Class using TensorFlow
# Multiple layers can be constructed by passing
#  n_nodes: number of layers in each layer as a list
#  actf: activation fuction in each layer
#  dropout: True or False in a list that designates whether dropout exists in each layer.
#  training_epochs: the number of training iteration steps
#  learning_rate: the learning rate of gradient descent algorithm
#  random_seed: the random seed number for reproducing the results. 

    def __init__(self, sess, n_samples, n_features, n_nodes=[10,5,2], actf=[tf.nn.relu, tf.nn.relu, tf.nn.sigmoid], \
                 dropout = [True, True, False],\
                 training_epochs=1000, learning_rate=0.001,\
                 batch_size=128, random_seed = 2):
        self.sess = sess
        self.n_samples = n_samples
        tf.set_random_seed(random_seed)
        self.X = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)
        self.batch_size = batch_size
        self.n_nodes = [n_features] + n_nodes
        self.nlayers = len(self.n_nodes) #number of layers including input and hidden layers
        self.dropout = dropout
        self.training_epochs = training_epochs
        self.hidden_layer = []
        self.actf = actf

        # set the weights/biases in the hidden layer
        if self.nlayers >= 2:
            for i in range(self.nlayers-1):
                self.hidden_layer.append({'weights':tf.Variable(tf.truncated_normal([self.n_nodes[i], self.n_nodes[i+1]], stddev=0.05)),
                               'biases':tf.Variable(tf.truncated_normal([self.n_nodes[i+1]], stddev=0.05))})

        # set the weights/biases in the output layer
        self.output_layer = {'weights': tf.Variable(tf.truncated_normal([self.n_nodes[-1], 2], stddev=0.05)),
                             'biases': tf.Variable(tf.truncated_normal([2], stddev=0.05))}

        # keep probability of drop-out.
        self.keep_prob = tf.placeholder(tf.float32)

        # calculate prediction given features X
        self.y_pred = self.MLP_model(self.X)

        # calculate cost
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_pred))

        # set the optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

    def MLP_model(self, X):
        #Create multi level perceptron model (feed forward neural network)
        l = X
        for i in range(self.nlayers-1):
            l = tf.add(tf.matmul(l, self.hidden_layer[i]['weights']), self.hidden_layer[i]['biases'])
            l = self.actf[i](l)
            if self.dropout[i]:
                l = tf.nn.dropout(l, self.keep_prob)

        output = tf.matmul(l, self.output_layer['weights']) + self.output_layer['biases']
        return output 

    def train_and_test(self, X_train, y_train, X_test, y_test, keep_prob=1.0):
        #input
        #  X_train: training features
        #  y_train: training target
        #  X_test: testing features
        #  y_test: testing target
        #  keep_prob: the keep probability for drop-out
        
        tf.global_variables_initializer().run(session=self.sess)
        cost_list = []
        test_cost_list = []

        # for each epoch
        for epoch in range(self.training_epochs):

            # calculate the number of data in each batch
            n_batch = int(self.n_samples / self.batch_size)

            # for each batch
            for i in range(n_batch):

                # calculate the offset and calculate features and target
                offset = (i * self.batch_size) % self.n_samples
                X_batch = X_train[offset:offset + self.batch_size, :]
                y_batch = y_train[offset:offset + self.batch_size]

                # run the optimizer
                self.sess.run(self.optimizer, feed_dict={self.X: X_batch, self.y: y_batch, self.keep_prob:keep_prob})

            # the last remaining data for the last batch
            if n_batch < self.n_samples:

                # calculate the offset and calculate features and target
                offset = n_batch * self.batch_size
                X_batch = X_train[offset:, :]
                y_batch = y_train[offset:]

                # run the optimizer
                self.sess.run(self.optimizer, feed_dict={self.X: X_batch, self.y: y_batch, self.keep_prob:keep_prob})

            # calculate the cost for training data set
            cost = self.sess.run(self.cost, feed_dict={self.X: X_train, self.y: y_train, self.keep_prob:keep_prob})

            # calculate the cost for testing data set
            test_cost = self.sess.run(self.cost, feed_dict={self.X: X_test, self.y: y_test, self.keep_prob:keep_prob})

            cost_list.append(cost)
            test_cost_list.append(test_cost)

            # Print informations at every 100 steps.
            if epoch % 100 == 0:
                sys.stdout.write("\repoch = %d, cost = %f, test_cost = %f" % (epoch, cost, test_cost))

        # output: prediction from testing features, training cost in a list, testing cost in a list
        return self.test(X_test), cost_list, test_cost_list

    #predict from testing data
    def test(self, X_test):
        if type(X_test) == pd.core.frame.DataFrame:
            X_test = X_test.values
        # For testing time, we need to keep drop-out to 0. In other words, keep_prob is 1.0
        return self.sess.run(self.MLP_model(X_test.astype(np.float32)), feed_dict={self.keep_prob:1.0})

    #calculate accuracy
    def accuracy(self, X_test, y_test, type="mean_absolute_error"):
        if type == "r2_score":
            acc = r2_score(y_test, self.query(X_test))
        elif type == "mean_absolute_error":
            acc = mean_absolute_error(y_test, self.query(X_test))
        return acc

