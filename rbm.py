# -*- coding: utf-8 -*-
"""
@created on: 2019-03-26,
@author: Vivek A Gupta,

Description:

..todo::
"""

import tensorflow as tf
from pre_processing import get_data, divide_batches, get_inference_data

# Data Paths.
train_path = "ml-100k/u1.base"
test_path = "ml-100k/u1.test"

# Model Params.
lr = 0.01
epochs = 1
batch_size = 100
nh = 600
k = 1

temp_train, test_set = get_data(train_path, test_path)
infer_train, infer_test = get_inference_data(test_set, temp_train)
training_set = divide_batches(temp_train, batch_size)
infer_train = divide_batches(infer_train, batch_size)
infer_test = divide_batches(infer_test, batch_size)


train_batch_length = len(training_set)
infer_batch_length = len(infer_test)


class RBM:
    def __init__(self, nv, nh, k):
        self.w = tf.Variable(tf.initializers.random_normal().__call__(shape=[nv, nh]))  # Weights
        self.vb = tf.Variable(tf.initializers.zeros().__call__(shape=[nv]))  # Visible Layer Bias
        self.hb = tf.Variable(tf.initializers.zeros().__call__(shape=[nh]))  # Hidden Layer Bias
        self.k = k

    @staticmethod
    def sample_prob(p):
        # return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))
        return tf.where(tf.less(p, tf.random_uniform(tf.shape(p), minval=0.0, maxval=1.0)),
                        x=tf.zeros_like(p),
                        y=tf.ones_like(p))

    def _sample_h(self, v):
        p_h_v = tf.nn.sigmoid(tf.matmul(v, self.w) + self.hb)
        h_ = self.sample_prob(p_h_v)
        return p_h_v, h_

    def _sample_v(self, h):
        p_v_h = tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.w)) + self.vb)
        v_ = self.sample_prob(p_v_h)
        return p_v_h, v_

    # Calculate Gradients.
    @staticmethod
    def calc_gradients(v0, ph0, vk, phk):
        positive_grad = tf.matmul(tf.transpose(v0), ph0)
        negative_grad = tf.matmul(tf.transpose(vk), phk)
        w_grad = lr * (positive_grad - negative_grad) / tf.to_float(
            tf.shape(v0)[0])  # Weight Gradient
        vb_grad = lr * tf.reduce_mean(v0 - vk, 0)  # Visible Bias Gradient
        hb_grad = lr * tf.reduce_mean(ph0 - phk, 0)  # Hidden Bias Gradient
        return w_grad, vb_grad, hb_grad

    # Update Weights and Biases.
    def update_wb(self, w_grad, v_grad, h_grad):
        self.w = tf.assign_add(self.w, w_grad)
        self.vb = tf.assign_add(self.vb, v_grad)
        self.hb = tf.assign_add(self.hb, h_grad)
        return [self.w, self.vb, self.hb]

    # Prediction.
    def prediction(self, v):
        ph0, h0 = self._sample_h(v)
        _, vk = self._sample_v(h0)
        vk = tf.where(tf.less(v, 0), v, vk)
        return vk

    @staticmethod
    def mask(labels, predictions):
        mask = tf.greater(labels, -1)
        labels = tf.boolean_mask(labels, mask)
        predictions = tf.boolean_mask(predictions, mask)
        return labels, predictions

    # Accuracy.
    @staticmethod
    def accuracy(labels, predictions):
        return tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Loss.
    @staticmethod
    def loss(labels, predictions):
        return tf.losses.mean_squared_error(labels=labels, predictions=predictions)

    # Train RBM.
    def train(self, v):
        ph0, h0 = self._sample_h(v)
        _, vk = self._sample_v(h0)
        vk = tf.where(tf.less(v, 0), v, vk)
        phk, hk = self._sample_h(vk)

        for i in range(1, self.k):
            _,vk = self._sample_v(hk)
            vk = tf.where(tf.less(v, 0), v, vk)
            phk, hk = self._sample_h(vk)

        w_grad, vb_grad, hb_grad = self.calc_gradients(v, ph0, vk, phk)
        grads = self.update_wb(w_grad, vb_grad, hb_grad)
        labels, predictions = self.mask(v, vk)
        train_acc = self.accuracy(labels, predictions)
        train_loss = self.loss(labels, predictions)
        return train_loss, train_acc

    # Test RBM.
    def test(self, v, v_test):
        pht, ht = self._sample_h(v)
        pvt, vt = self._sample_v(ht)
        vt = tf.where(tf.less(v_test, 0), v_test, vt)
        labels, predictions = self.mask(v_test, vt)
        test_loss = self.loss(labels=labels, predictions=predictions)
        test_acc = self.accuracy(labels, predictions)
        return test_loss, test_acc


nv = len(training_set[0][0])

# Inputs
v = tf.placeholder(dtype=tf.float32, shape=[None, nv])  # Visible Layer
v_test = tf.placeholder(dtype=tf.float32, shape=[None, nv])  # Visible Layer

rbm_obj = RBM(nv, nh, k=k)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
tf.add_to_collection('v', v)
tf.add_to_collection('prediction', rbm_obj.prediction(v))

with tf.Session() as sess:
    sess.run(init)

    for i in range(epochs):
        tot_loss = 0
        tot_acc = 0
        print(f"\n*** Epoch: {i+1} ***\n")
        for data in training_set:
            l, a = sess.run(rbm_obj.train(v), feed_dict={v: data})
            tot_loss += l
            tot_acc += a
            print(f"Batch Loss: {l}\nBatch Accuracy: {a}")
        print(f"\n-----Epoch: {i+1}-----\nTrain Loss: {tot_loss / train_batch_length}\nTrain Accuracy: {tot_acc / train_batch_length}\n")
        saver.save(sess, 'rbm_models/model_1', global_step=i)

    tot_loss = 0
    tot_acc = 0
    for train_data, test_data in zip(infer_train, infer_test):
        l, a = sess.run(rbm_obj.test(v, v_test), feed_dict={v: train_data, v_test: test_data})
        tot_loss += l
        tot_acc += a
    print(f"-----INFERENCE-----\nInference Loss: {tot_loss / infer_batch_length}\nInference Accuracy: {tot_acc / infer_batch_length}\n")
