#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 20:50:38 2018

@author: jose
"""
import tensorflow as tf
import numpy as np

saver = tf.train.import_meta_graph('/home/jose/Templates/Pix2Pix/pix2pix-tensorflow_jose/my-model.meta')
#graph = tf.get_default_graph()
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
sess.graph.get_operations()
e1 = sess.graph.get_tensor_by_name('generator/g_e1_conv/w:0')
eb1 = sess.graph.get_tensor_by_name('generator/g_e1_conv/biases:0')
we1 = e1.eval(session=sess)
eb1 = eb1.eval(session=sess)
We1 = [we1, eb1]
e2 = sess.graph.get_tensor_by_name('generator/g_e2_conv/w:0')
eb2 = sess.graph.get_tensor_by_name('generator/g_e2_conv/biases:0')
print e2.eval(session=sess).shape
e3 = sess.graph.get_tensor_by_name('generator/g_e3_conv/w:0')
eb3 = sess.graph.get_tensor_by_name('generator/g_e3_conv/biases:0')
#print e3.eval(session=sess).shape
e4 = sess.graph.get_tensor_by_name('generator/g_e4_conv/w:0')
eb4 = sess.graph.get_tensor_by_name('generator/g_e4_conv/biases:0')
#print e4.eval(session=sess).shape
e5 = sess.graph.get_tensor_by_name('generator/g_e5_conv/w:0')
eb5 = sess.graph.get_tensor_by_name('generator/g_e5_conv/biases:0')
#print e5.eval(session=sess).shape
e6 = sess.graph.get_tensor_by_name('generator/g_e6_conv/w:0')
eb6 = sess.graph.get_tensor_by_name('generator/g_e6_conv/biases:0')
#print e6.eval(session=sess).shape
e7 = sess.graph.get_tensor_by_name('generator/g_e7_conv/w:0')
eb7 = sess.graph.get_tensor_by_name('generator/g_e7_conv/biases:0')
#print e7.eval(session=sess).shape
e8 = sess.graph.get_tensor_by_name('generator/g_e8_conv/w:0')
eb8 = sess.graph.get_tensor_by_name('generator/g_e8_conv/biases:0')
#print e8.eval(session=sess).shape
# graph.get_operations()

d1 = sess.graph.get_tensor_by_name('generator/g_d1/w:0')
db1 = sess.graph.get_tensor_by_name('generator/g_d1/biases:0')
#print d1.eval(session=sess).shape
#print db1.eval(session=sess).shape
d2 = sess.graph.get_tensor_by_name('generator/g_d2/w:0')
db2 = sess.graph.get_tensor_by_name('generator/g_d2/biases:0')
#print d2.eval(session=sess).shape
d3 = sess.graph.get_tensor_by_name('generator/g_d3/w:0')
db3 = sess.graph.get_tensor_by_name('generator/g_d3/biases:0')
#print d3.eval(session=sess).shape
d4 = sess.graph.get_tensor_by_name('generator/g_d4/w:0')
db4 = sess.graph.get_tensor_by_name('generator/g_d4/biases:0')
#print d4.eval(session=sess).shape
d5 = sess.graph.get_tensor_by_name('generator/g_d5/w:0')
db5 = sess.graph.get_tensor_by_name('generator/g_d5/biases:0')
#print d5.eval(session=sess).shape
d6 = sess.graph.get_tensor_by_name('generator/g_d6/w:0')
db6 = sess.graph.get_tensor_by_name('generator/g_d6/biases:0')
#print d6.eval(session=sess).shape
d7 = sess.graph.get_tensor_by_name('generator/g_d7/w:0')
db7 = sess.graph.get_tensor_by_name('generator/g_d7/biases:0')
#print d7.eval(session=sess).shape
d8 = sess.graph.get_tensor_by_name('generator/g_d8/w:0')
db8 = sess.graph.get_tensor_by_name('generator/g_d8/biases:0')
print d8.eval(session=sess).shape
sess.close()
