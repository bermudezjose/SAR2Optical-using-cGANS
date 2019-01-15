import tensorflow as tf
# from model_semiautomatic import pix2pix
import numpy as np
from ops import *
import keras
mnist = tf.keras.datasets.mnist

class certified(object):
	def __init__(self): 

		(x_train, y_train),(x_test, y_test) = mnist.load_data()
		x_train, x_test = x_train / 255.0, x_test / 255.0
		self.x_train = x_train
		self.x_test = x_test
		self.y_train = y_train
		self.y_test = y_test

		self.n_trn_samples = x_train.shape[0]
		self.n_tst_samples = x_test.shape[0]
		self.dim = x_train.shape[1]
		self.n_features = self.dim * self.dim
		self.n_classes = 10
		self.n_hidden = 512
		self.batch_size = 32
		self.build_model()

	def build_model(self):

		self.data_classifier = tf.placeholder(tf.float32,
                                              [None, self.n_features], name='data_classifier')
		self.labels_classifier = tf.placeholder(tf.float32,
			                                    [None, self.n_classes], name='labels_classifier')
		self.dropout_keep_prob = tf.placeholder(tf.float32)

		self.C, self.C_logits = self.classifier(self.data_classifier, self.dropout_keep_prob)

		self.c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.C_logits, labels=self.labels_classifier))

		t_vars = tf.trainable_variables()
		self.c_vars = [var for var in t_vars if 'c_' in var.name]
		self.saver = tf.train.Saver()

	def classifier(self, features, dropout_keep_prob):
		with tf.variable_scope("classifier") as scope:
			out_layer = mlp(input=features, n_features=self.n_features, n_classes=self.n_classes, n_hidden=self.n_hidden, dropout_keep_prob=dropout_keep_prob)
			return tf.nn.softmax(out_layer), out_layer

	def train(self):
		index = range(self.n_trn_samples)
		x_train = np.float32(self.x_train.reshape(self.n_trn_samples, self.dim * self.dim))
		x_test = np.float32(self.x_test.reshape(self.n_tst_samples, self.dim * self.dim))
		y_train_one = keras.utils.to_categorical(self.y_train, self.n_classes)
		y_test_one = keras.utils.to_categorical(self.y_test, self.n_classes)

		with tf.Session() as sess:
			c_optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.c_loss, var_list=self.c_vars)
			init_op = tf.global_variables_initializer()
			sess.run(init_op)
			self.writer = tf.summary.FileWriter("./logs", sess.graph)

			for epoch in xrange(100):
				loss = []
				batch_idxs = self.n_trn_samples // self.batch_size
				np.random.shuffle(index)
				data = x_train.copy()
				labels = y_train_one.copy()
				data = data[index]
				labels = labels[index]
				for idx in xrange(0, batch_idxs):
					trn_samples = data[idx * self.batch_size:(idx + 1) * self.batch_size]
					trn_labels = labels[idx * self.batch_size:(idx + 1) * self.batch_size]

					_, c_loss = sess.run([c_optim, self.c_loss],
						feed_dict={self.data_classifier: trn_samples, self.labels_classifier: trn_labels, self.dropout_keep_prob: 0.8})
					loss.append(c_loss)

				print ("epoch --->", epoch, "loss --->", np.mean(loss))
			# pred = self.c_loss.eval()({self.data_classifier: trn_samples})
			pred = sess.run([self.C],
									feed_dict={self.data_classifier: x_test, self.dropout_keep_prob: 1.0})
			np_pred = np.array(pred)
			np_pred = np_pred.reshape(self.n_tst_samples, self.n_classes)
			predictions = np.argmax(np_pred, axis=1)
			# (x_train, y_train),(x_test, y_test) = mnist.load_data()
			# x_train, x_test = x_train / 255.0, x_test / 255.0
			print(np.sum(predictions == self.y_test) / np.float32(self.n_tst_samples))
			# print (np.shape(pred))
			# pred = np.array(pred).reshape(self.n_trn_samples, self.n_classes)
			# # print (pred[:10])
			# pred = np.argmax(pred, axis=1)
			# pred = np.float32(pred.reshape(self.n_trn_samples))
			# print (pred.dtype)
			# l = np.float32(self.y_train.copy())
			# print (self.y_train)
			# # oa = np.sum(pre == self.y_test) / np.float32(self.n_tst_samples)
			# oa = pre + l
			# # oa = np.equal(pre, self.train)
			# print (oa)

		return pred

model = certified()
pred = model.train()







# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(512, activation=tf.nn.relu),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=5)
# print (model.evaluate(x_test, y_test))