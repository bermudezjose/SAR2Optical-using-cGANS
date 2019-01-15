import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.decomposition import PCA
from utils import *
from sklearn.manifold import TSNE
# %matplotlib inline

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123

# Utility function to visualize the outputs of PCA and t-SNE

def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.show(block=False)
    return f, ax, sc, txts



time_start = time.time()


opt_img_name = '20160505/'
opt_root_patch = '/mnt/Data/DataBases/RS/SAR/Campo Verde/LANDSAT/'
image_path = opt_root_patch + opt_img_name
labels_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/Labels_uint8/10_May_2016.tif'
# trn_Data, trn_Labels, labels2new_labels = create_training_samples_Classifier_oneClass(image_path, labels_path)
trn_data, trn_labels, tst_data, tst_labels = load_train_test_samples(image_path, labels_path)

pca = PCA(n_components=4)
pca_model = pca.fit(trn_data)
pca_result_trn = pca_model.transform(trn_data)
pca_result_tst = pca_model.transform(tst_data)
print(np.shape(pca_result_trn))
print(np.shape(pca_result_tst))

print 'PCA done! Time elapsed: {} seconds'.format(time.time()-time_start)

pca_trn_df = pd.DataFrame(columns = ['pca1','pca2','pca3','pca4'])
pca_tst_df = pd.DataFrame(columns = ['pca1','pca2','pca3','pca4'])

pca_trn_df['pca1'] = pca_result_trn[:,0]
pca_trn_df['pca2'] = pca_result_trn[:,1]
pca_trn_df['pca3'] = pca_result_trn[:,2]
pca_trn_df['pca4'] = pca_result_trn[:,3]

pca_tst_df['pca1'] = pca_result_tst[:,0]
pca_tst_df['pca2'] = pca_result_tst[:,1]
pca_tst_df['pca3'] = pca_result_tst[:,2]
pca_tst_df['pca4'] = pca_result_tst[:,3]

print 'Variance explained per principal component: {}'.format(pca.explained_variance_ratio_)

top_two_comp_trn = pca_trn_df[['pca1','pca2']] # taking first and second principal component
top_two_comp_tst = pca_tst_df[['pca1','pca2']] # taking first and second principal component

fashion_scatter(top_two_comp_trn.values, trn_labels) # Visualizing the PCA output
fashion_scatter(top_two_comp_tst.values, tst_labels)
plt.pause(10000)

# time_start = time.time()

# fashion_tsne = TSNE(random_state=RS).fit_transform(trn_Data)

# print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

# fashion_scatter(fashion_tsne,trn_Labels)