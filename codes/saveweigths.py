#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 10:34:17 2018

@author: jose
"""
import numpy as np
def save_weights(sess):

    weigths = {}
    sess.graph.get_operations()
    e1 = sess.graph.get_tensor_by_name('generator/g_e1_conv/w:0')
    eb1 = sess.graph.get_tensor_by_name('generator/g_e1_conv/biases:0')
    w = e1.eval(session=sess)
    b = eb1.eval(session=sess)
    We1 = [w, b]
    weigths["We1"] = We1
    e2 = sess.graph.get_tensor_by_name('generator/g_e2_conv/w:0')
    eb2 = sess.graph.get_tensor_by_name('generator/g_e2_conv/biases:0')
    w = e2.eval(session=sess)
    b = eb2.eval(session=sess)
    We2 = [w, b]
    weigths["We2"] = We2
    #print e2.eval(session=sess).shape
    e3 = sess.graph.get_tensor_by_name('generator/g_e3_conv/w:0')
    eb3 = sess.graph.get_tensor_by_name('generator/g_e3_conv/biases:0')
    w = e3.eval(session=sess)
    b = eb3.eval(session=sess)
    We3 = [w, b]
    weigths["We3"] = We3
    #print e3.eval(session=sess).shape
    e4 = sess.graph.get_tensor_by_name('generator/g_e4_conv/w:0')
    eb4 = sess.graph.get_tensor_by_name('generator/g_e4_conv/biases:0')
    w = e4.eval(session=sess)
    b = eb4.eval(session=sess)
    We4 = [w, b]
    weigths["We4"] = We4
    #print e4.eval(session=sess).shape
    e5 = sess.graph.get_tensor_by_name('generator/g_e5_conv/w:0')
    eb5 = sess.graph.get_tensor_by_name('generator/g_e5_conv/biases:0')
    w = e5.eval(session=sess)
    b = eb5.eval(session=sess)
    We5 = [w, b]
    weigths["We5"] = We5
    #print e5.eval(session=sess).shape
    e6 = sess.graph.get_tensor_by_name('generator/g_e6_conv/w:0')
    eb6 = sess.graph.get_tensor_by_name('generator/g_e6_conv/biases:0')
    w = e6.eval(session=sess)
    b = eb6.eval(session=sess)
    We6 = [w, b]
    weigths["We6"] = We6
    #print e6.eval(session=sess).shape
    e7 = sess.graph.get_tensor_by_name('generator/g_e7_conv/w:0')
    eb7 = sess.graph.get_tensor_by_name('generator/g_e7_conv/biases:0')
    w = e7.eval(session=sess)
    b = eb7.eval(session=sess)
    We7 = [w, b]
    weigths["We7"] = We7
    #print e7.eval(session=sess).shape
    e8 = sess.graph.get_tensor_by_name('generator/g_e8_conv/w:0')
    eb8 = sess.graph.get_tensor_by_name('generator/g_e8_conv/biases:0')
    w = e8.eval(session=sess)
    b = eb8.eval(session=sess)
    We8 = [w, b]
    weigths["We8"] = We8
    #print e8.eval(session=sess).shape
    # graph.get_operations()

    d1 = sess.graph.get_tensor_by_name('generator/g_d1/w:0')
    db1 = sess.graph.get_tensor_by_name('generator/g_d1/biases:0')
    w = d1.eval(session=sess)
    b = db1.eval(session=sess)
    Wb1 = [w, b]
    weigths["Wb1"] = Wb1
    # print d1.eval(session=sess).shape
    # print db1.eval(session=sess).shape
    d2 = sess.graph.get_tensor_by_name('generator/g_d2/w:0')
    db2 = sess.graph.get_tensor_by_name('generator/g_d2/biases:0')
    w = d2.eval(session=sess)
    b = db2.eval(session=sess)
    Wb2 = [w, b]
    weigths["Wb2"] = Wb2
    # print d2.eval(session=sess).shape
    d3 = sess.graph.get_tensor_by_name('generator/g_d3/w:0')
    db3 = sess.graph.get_tensor_by_name('generator/g_d3/biases:0')
    w = d3.eval(session=sess)
    b = db3.eval(session=sess)
    Wb3 = [w, b]
    weigths["Wb3"] = Wb3
    # print d3.eval(session=sess).shape
    d4 = sess.graph.get_tensor_by_name('generator/g_d4/w:0')
    db4 = sess.graph.get_tensor_by_name('generator/g_d4/biases:0')
    w = d4.eval(session=sess)
    b = db4.eval(session=sess)
    Wb4 = [w, b]
    weigths["Wb4"] = Wb4
    # print d4.eval(session=sess).shape
    d5 = sess.graph.get_tensor_by_name('generator/g_d5/w:0')
    db5 = sess.graph.get_tensor_by_name('generator/g_d5/biases:0')
    w = d5.eval(session=sess)
    b = db5.eval(session=sess)
    Wb5 = [w, b]
    weigths["Wb5"] = Wb5
    # print d5.eval(session=sess).shape
    d6 = sess.graph.get_tensor_by_name('generator/g_d6/w:0')
    db6 = sess.graph.get_tensor_by_name('generator/g_d6/biases:0')
    w = d6.eval(session=sess)
    b = db6.eval(session=sess)
    Wb6 = [w, b]
    weigths["Wb6"] = Wb6
    # print d6.eval(session=sess).shape
    d7 = sess.graph.get_tensor_by_name('generator/g_d7/w:0')
    db7 = sess.graph.get_tensor_by_name('generator/g_d7/biases:0')
    w = d7.eval(session=sess)
    b = db7.eval(session=sess)
    Wb7 = [w, b]
    weigths["Wb7"] = Wb7
    # print d7.eval(session=sess).shape
    d8 = sess.graph.get_tensor_by_name('generator/g_d8/w:0')
    db8 = sess.graph.get_tensor_by_name('generator/g_d8/biases:0')
    w = d8.eval(session=sess)
    b = db8.eval(session=sess)
    Wb8 = [w, b]
    weigths["Wb8"] = Wb8
    np.save("weigths", weigths)
