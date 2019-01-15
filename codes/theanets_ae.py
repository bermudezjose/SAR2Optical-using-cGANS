import theanets
import random as rng
from sklearn import preprocessing as pre
import numpy as np
 
def learnFeatures(data,
                 encoding_dim = 32,
                 activation_function='sigmoid',
                 output_function = 'sigmoid',
                 kSize = 3,
                 NChannelsPerImage = 3,
                 lamda_act = (10e-2,10e-3),
                 lamda_w= (10e-2,10e-3),
                 nb_epoch = 50 ,
                 batch_size=256):
    
    N = kSize**2*NChannelsPerImage
    
    ae = theanets.Autoencoder([N, (encoding_dim, 'sigmoid'), (N, 'tied')])
    ae.train([data], hidden_l1 = lamda_act)                
    
    weights = ae.find('hid1', 'w').get_value()
    bias = ae.find('hid1', 'b').get_value()

    
    weights = encoder.get_weights()[0].T
    weights= np.reshape(weights,(encoding_dim, NChannelsPerImage, kSize, kSize))
    
    return [weights, bias]

        
        
        