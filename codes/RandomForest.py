
from PIL import Image
import numpy as np
import theano as th
import glob
from csv import reader, writer
from libtiff import *
import sys
import cv2
import random 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from unsupervisedlearning import UnsupervisedLearning


def loadImagesMat(ImageSize, NChannelsPerImage):
	# cargar las imagenes de las direcciones que tengo almacenas en los csv
    path_images = list(reader(open('file_imageXdata.csv','r')))
    path_labels = list(reader(open('file_labelXdata.csv','r')))

    imagesData = []
    imagesLabel = []
    cont = 0

    for f in path_images:
        tif = TIFF.open(f[0], 'r')
        tif = tif.read_image()
        #tif = np.transpose(tif)
        imagesData.append(tif)
                

    nImages = len(path_images)/NChannelsPerImage
    temp_image = np.zeros(shape = (nImages, NChannelsPerImage,ImageSize[0],ImageSize[1]), dtype='int16')
    
	# salvar en un array de la forma (nImages, NChannelsPerImage,ImageSize[0],ImageSize[1])
    for i in range(0,nImages):
        for j in range(0,NChannelsPerImage):
            temp_image[i,j,:,:] = imagesData[cont]
            cont += 1

	# reshape (nImages, NChannelsPerImage, ImageSize[0]*ImageSize[1])
    learningImgs = np.asarray(temp_image)
    learningImgs = learningImgs.reshape(nImages, NChannelsPerImage, ImageSize[0]*ImageSize[1])
	
	# cargar las imagenes de labels y hacer lo mismo
    for f in path_labels:
        tif = TIFF.open(f[0], 'r')
        tif = tif.read_image()
        #tif = np.transpose(tif)
        imagesLabel.append(tif)
    
    learningLabels = np.asarray(imagesLabel)
    learningLabels = learningLabels.reshape(nImages, ImageSize[0]*ImageSize[1])

    print ("Tensor de Imagenes cargadas ---> ", learningImgs.shape)
    print ("Tensor de Imagenes cargadas, labels ---> ", learningLabels.shape)
    return [learningImgs,learningLabels]
	
def pixelExtractor(imgs,labelmap, numGroups):

	unique_labels = np.unique(labelmap)
	NChannelsPerImage = imgs.shape[0]
	xy_size = imgs.shape[1]
	input_group = imgs.T
	input_label = labelmap.T
   
	# encontrar todos los pixeles distintos de 0 en la matrix de labels
	coords = zip(*np.where(labelmap != 0))
	# obtener numero de pixels por grupo
	len_group = len(coords)/numGroups
	
	# crear matrices para guardar los gruos y los labels de los grupos
	mat_temp = np.zeros(shape = (numGroups,len_group,NChannelsPerImage))
	label_temp = np.zeros(shape = (numGroups,len_group,1))

	# tomar los subconjutnos de coordenadas para cada grupo, si en alguno
	# no estan representada todas las calses hago shuffle de las coordenadas
	# y comienzo de nuevo
	left_idx = 0
	idx_loops = 1
	while (idx_loops):
		cont = 0
		for k in range(numGroups):
			lab_group = np.take(input_label, coords[left_idx:left_idx+len_group])
			diff_labels = np.unique(lab_group)
			if len(diff_labels) == (len(unique_labels)-1):
				# si estan todas las clases guardo los piexels y los labels
				mat_temp[k,:,:] = np.take(input_group,coords[left_idx:left_idx+len_group])
				label_temp[k,:] = lab_group
				left_idx = left_idx+len_group
				cont = cont + 1
				if cont == numGroups:
					idx_loops = 0
			else:
				random.shuffle(coords)
				left_idx = 0
				break;
				
	mat_group = np.array(mat_temp)
	label_group = np.array(label_temp)

	return  [mat_group,label_group]
	
	
def random_forest(data,responses,n_trees,max_depth):
    '''
    Auto trains an OpenCV SVM.
    '''
    data = np.float32(data)
    responses = np.float32(responses) 
    params = dict(max_depth = max_depth, max_num_of_trees_in_the_forest=n_trees,termcrit_type=cv2.TERM_CRITERIA_MAX_ITER)
    #params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_EPS_SVR , p=1.0 )
    model = cv2.RTrees()
    model.train(data,cv2.CV_ROW_SAMPLE,responses,params=params)
    return model
    

def segm_image (pred_labels,img_control):
    # funcion para la reconstruccion
    temp_copy = img_control;
    verts = zip(*np.where(img_control != 0))
    temp_copy[verts] = 1
    segment_img = temp_copy
    
    for i in range(len(pred_labels)):
        segment_img[verts[i]] = pred_labels[i]
    
    segment_img = segment_img.reshape(ImageSize[0],ImageSize[1])
    
    return segment_img


if __name__ == '__main__':
    
    ImageSize = (948, 1068)
    NChannelsPerImage = 6
    Groups = 3
    n_trees=250
    max_depth=25
    nImages = 1 # Dejar fijo
    ImgNumber = 0 # Imagen selecionada
    depth = 128
    kSize = 7
    stride = 1
    learning_rate = (10e-2,10e-3)
    w_decay = (0, 10e-5)
    w_lasso = (0, 10e-5)
    sparse_reg = 10e-5
    nb_epoch = 100    
    
    random.seed(1)
	
    [learningImgs,learningLabels] = loadImagesMat(ImageSize,NChannelsPerImage)
    print 'Load data matrix and label matrix ok\n'
    	
     #extraer los atributos solamente de la imagen 0 para clasificar
    #    [data,labels] = pixelExtractor(learningImgs[0],learningLabels[0],Groups)
    #    print ('Extract feacture ok\n')
       
      
    
    #xy_size = imgs.shape[1]
    input_group = np.float32(learningImgs[ImgNumber])/65535
    input_label = learningLabels[ImgNumber]
    unique_labels = np.unique(input_label)
    print unique_labels
    NChannelsPerImage = input_group.shape[0]
    print ('Dimensiones de la imagen de entrada ----> ', input_group.shape)    
    
    input_group = input_group.reshape(1, NChannelsPerImage, ImageSize[0], ImageSize[1])
#    unsupervisedModel = UnsupervisedLearning()
#    featuremap = unsupervisedModel.featureExtraction(input_group,
#                                                     nImages,
#                                                     NChannelsPerImage,
#                                                     nlayers=1,
#                                                     depth=depth,
#                                                     kSize=(kSize,3),
#                                                     nPatches=stride,
#                                                     pooling_size=(1,1),
#                                                     learning_rate = learning_rate,
#                                                     w_decay = w_decay,
#                                                     w_lasso = w_lasso,
#                                                     sparse_reg = sparse_reg,
#                                                     nb_epoch= nb_epoch
#                                                     )
    featuremap = input_group.reshape(NChannelsPerImage, ImageSize[0] * ImageSize[1])
    featuremap = featuremap.T
    print ("Dimensiones de los atributos extraidos --->", featuremap.shape)
         
    numGroups = Groups
    	# encontrar todos los pixeles distintos de 0 en la matrix de labels
    coords = zip( *np.where(learningLabels[0] != 0) )
    # obtener numero de pixels por grupo
    len_group = len(coords)/numGroups
    #	
    	# crear matrices para guardar los gruos y los labels de los grupos
    mat_temp = np.zeros(shape = (numGroups,len_group,NChannelsPerImage))
    label_temp = np.zeros(shape = (numGroups,len_group,1))
    
    groups = range(numGroups)
    idx = np.repeat(groups,len_group+1)
    idx = idx[0:len(coords)]
    random.shuffle(idx)
    data = featuremap[coords]
    data = data.reshape(len(coords), NChannelsPerImage)
    labels = input_label[coords]
    predict_data = [] ## corregir esto despues para que sea mas rapido
    
    #for save labels of gropus, codigo agregado
    label_seg = []
    # fin
    
    true_labels = []
    wrong_class = 0.0
    correct_class = 0.0
    false_positives = []
    count_fp = 0
    
    
    for i in range(numGroups):
         print 'Start training\n'
         # elimino el grupo i del conjutno de entrenamiento y hago un union de los grupos restantes
         for_seg = labels
         trainData = data[idx != i]
         testData = data[idx == i]
         
         trainLabels = labels[idx != i]
         testLabels = labels[idx == i]
         
         trainData = trainData
         trainLabels = np.float32(trainLabels)
         
         # training
         model = random_forest(trainData,trainLabels,n_trees,max_depth)
         
         # predict
         testLabels = np.float32(testLabels)
         
         cont = 0
         for f in testData:
             result = model.predict(f)
             predict_data.append(result)
             label_seg.append(result)
             true_labels.append(testLabels[cont])
             if result != testLabels[cont] :
                 wrong_class += 1
                 #false_positives [count_fp] = np.int8(predict_data[cont])
                 #count_fp += 1
             else:
                 correct_class += 1
             cont += 1
         
         conf_matrix = confusion_matrix(true_labels, predict_data) 
         
         # codigo agregado para calcular accuracy y hacer reconst
         acc_result = accuracy_score(true_labels, predict_data)
         
         label_seg = np.asarray(label_seg)
         label_seg = label_seg.reshape(len(label_seg),1)

         for_seg[idx == i] = label_seg
         label_seg = []
         # fin del codigo agregado
         
         #print (np.int8(100*np.float32(conf_matrix)/np.sum(conf_matrix, axis=1)))
         print ('Confusion Matrix ....')
         print (conf_matrix)
         
         print ('Classes correctamente classificadas ---->', 100 * correct_class/len(true_labels))
         print ('Classes classificadas incorrectamente ---->',100 * wrong_class/len(true_labels))
         
         
    # reconstruction section
    final_image = segm_image (for_seg,learningLabels[ImgNumber])
    plt.imshow(final_image,'spectral')
    plt.show()



         
#    print ('Configurations used .....')
#    print ('max_depth =', max_depth, 'nImages = ', nImages, 'encodingDim = ', depth, 'kSize = ', kSize,
#           'stride = ', stride, 'learning_rate = ', learning_rate, 'w_decay = ', w_decay, 'w_lasso = ', w_lasso)
   

#    for i in range(Groups):
#         print 'Start training\n'
#         # elimino el grupo i del conjutno de entrenamiento y hago un union de los grupos restantes
#         temp1 = np.delete(data, (i), axis=0)
#         trainData = np.rollaxis(temp1, 2, 1).reshape(temp1.shape[0]*temp1.shape[1], temp1.shape[2])
#         trainData = np.float32(trainData)/65535
#         temp2 = np.delete(labels, (i),axis=0)
#         responses = np.rollaxis(temp2, 0, 1).reshape(temp2.shape[0]*temp2.shape[1],1)
#         responses = np.float32(responses)
#         
#         # training
#         model = random_forest(trainData,responses,n_trees,max_depth)
#         		
#         # predict
#         test_data = np.float32(data[i])/65535
#         test_label = np.float32(labels[i])
#         predict_data =  np.zeros(shape = (test_data.shape[0]),dtype=np.float32)
#         
#         cont = 0
#         for f in test_data:
#             predict_data[cont] = model.predict(f)
#             cont = cont + 1
#		
#         # confusion
#         print predict_data.shape
#         conf_matrix = confusion_matrix(test_label, predict_data)
#		
#         print conf_matrix
		


