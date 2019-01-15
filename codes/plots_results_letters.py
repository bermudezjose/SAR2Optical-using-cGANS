import pandas as pd
#from matplotlib import rcParams
#rcParams['font.family'] = "cursive"
#rcParams['font.sans-serif'] = ['Tahoma']
#import matplotlib.pyplot as plt
import numpy as np
#from matplotlib import rc
import matplotlib.pylab as plt

#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)

# Overall Acc
#raw_data = {'Images': ['13DEC15', '18MAR16', '05MAY16', '08JUL16', '24JUL16'],
#            'Real': [85.9, 76.3, 85.3, 75.6, 70.4],
#            'Fake': [77.1, 60.4, 77.4, 64.1, 65.3],
#            'SAR': [54.3, 46.8, 51.6, 39.0, 30.2]}
# TST AREA
#raw_data = {'Images': ['13DEC15', '18MAR16', '05MAY16', '08JUL16', '24JUL16'],
#            'Real': [83.1, 70.8, 84.6, 69.7, 65.3],
#            'Fake': [77.5, 54.0, 72.5, 55.6, 59.1],
#            'SAR': [57.1, 45.0, 58.0, 44.8, 29.7]}
raw_data = {'Images': ['$O_{a}$', '$S_{a}$', '$O_{b}$', '$G[S_{a}]$', '$G[S_{a}S_{b}]$', '$G[S_{a}O_{b}]$', '$G[S_{a}S_{b}O_{b}]$'],
            'OA':       [84.6, 58.0, 69.2, 68.8, 72.6, 73.6, 74.6],
            'F1-Score': [51.4, 20.2, 41.8, 35.6, 38.8, 46.3, 46.8]
            }
df = pd.DataFrame(raw_data, columns=['Images', 'OA', 'F1-Score'])
print df

# Setting the positions and width for the bars
pos = list(range(len(df['OA'])))
width = 0.35

# Plotting the bars
fig, ax = plt.subplots(figsize=(14, 5))

# Create a bar with pre_score data,
# in position pos,
plt.bar(pos,
        # using df['pre_score'] data,
        df['OA'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#0000F0',
        # with label the first value in first_name
        label=df['Images'][0],
        edgecolor='black',
        hatch=""
        )

# Create a bar with mid_score data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos],
        # using df['mid_score'] data,
        df['F1-Score'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#00F000',
        # with label the second value in first_name
        label=df['Images'][1],
        edgecolor='black',
        hatch="\/")

#, fontweight='bold'
# Set the y axis label
ax.set_ylabel(' Accuracy $\%$ ', fontsize=24, style='italic')

# Set the chart's title
# ax.set_title('Overall Acurracy and Average F1-Score', fontsize=28)

# Set the position of the x ticks
ax.set_xticks([p + 0.5 * width for p in pos])

# number above bars graphs
for p in ax.patches: ax.annotate(np.round(p.get_height(),decimals=2),
                                 (p.get_x()+p.get_width()/2., p.get_height()),
                                 ha='center', va='center', xytext=(0, 10), fontsize=20,
                                 textcoords='offset points')

# Set the labels for the x ticks
# ax.set_xticklabels(df['Images'], fontsize=24, style='italic')
ax.set_xticklabels(df['Images'], fontsize=27)

# Set the labels for the y ticks
ax.yaxis.set_tick_params(labelsize=20)

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*2)
plt.ylim([15, 100])
ax.set_rasterized(True)

# Adding the legend and showing the plot

plt.legend(['OA', 'F1-$Score$'], bbox_to_anchor=(0., 1.02, 1., .102), loc='upper center',
           ncol=2,  borderaxespad=0., fontsize=18)
# plt.legend(['OA', 'F1-$Score$'], loc='upper center', fontsize=15)
# plt.grid()
plt.show(block=False)
fig.tight_layout()
plt.savefig('/home/jose/Drive/PUC/Tesis/PropostaTesis/campo_verde_metrics.pdf', dpi=600)
#/home/jose/Drive/PUC/Tesis/PropostaTesis/
# plt.savefig('/home/jose/Drive/PUC/WorkPlace/ISSPRS/monotemporal_OA4.eps', dpi=100)

###############################################################################
raw_data = {'Images': ['$O_{a}$', '$S_{a}$', '$O_{b}$', '$G[S_{a}]$', '$G[S_{a}S_{b}]$', '$G[S_{a}O_{b}]$', '$G[S_{a}S_{b}O_{b}]$'],
            'OA':       [93.4, 74.6, 71.4, 83.1, 86.6, 91.7, 92.7],
            'F1-Score': [61.2, 47.4, 45.7, 50.8, 52.2, 54.8, 56.0]
            }
df = pd.DataFrame(raw_data, columns=['Images', 'OA', 'F1-Score'])
print df

# Setting the positions and width for the bars
pos = list(range(len(df['OA'])))
width = 0.35

# Plotting the bars
fig, ax = plt.subplots(figsize=(14, 5))

# Create a bar with pre_score data,
# in position pos,
plt.bar(pos,
        # using df['pre_score'] data,
        df['OA'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#0000F0',
        # with label the first value in first_name
        label=df['Images'][0],
        edgecolor='black',
        )

# Create a bar with mid_score data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos],
        # using df['mid_score'] data,
        df['F1-Score'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#00F000',
        # with label the second value in first_name
        label=df['Images'][1],
        edgecolor='black',
        hatch="\/")

#, fontweight='bold'
# Set the y axis label
ax.set_ylabel(' Accuracy $\%$ ', fontsize=24, style='italic')

# Set the chart's title
# ax.set_title('Overall Acurracy and Average F1-Score', fontsize=28)

# Set the position of the x ticks
ax.set_xticks([p + 0.5 * width for p in pos])

# number above bars graphs
for p in ax.patches: ax.annotate(np.round(p.get_height(),decimals=2),
                                 (p.get_x()+p.get_width()/2., p.get_height()),
                                 ha='center', va='center', xytext=(0, 10), fontsize=20,
                                 textcoords='offset points')

# Set the labels for the x ticks
# ax.set_xticklabels(df['Images'], fontsize=24, style='italic')
ax.set_xticklabels(df['Images'], fontsize=27)

# Set the labels for the y ticks
ax.yaxis.set_tick_params(labelsize=20)

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*2)
plt.ylim([15, 100])
ax.set_rasterized(True)

# Adding the legend and showing the plot

plt.legend(['OA', 'F1-$Score$'], bbox_to_anchor=(0., 1.02, 1., .102), loc='upper center',
           ncol=2,  borderaxespad=0., fontsize=18)
# plt.legend(['OA', 'F1-$Score$'], loc='upper center', fontsize=15)
# plt.grid()
plt.show(block=False)
fig.tight_layout()
plt.savefig('/home/jose/Drive/PUC/Tesis/PropostaTesis/rio_branco_metrics.pdf', dpi=600)

#
## ############################
## Average Acc
##raw_data = {'Images': ['13DEC15', '18MAR16', '05MAY16', '08JUL16', '24JUL16'],
##            'Real': [66.6, 70.3, 65.3, 66.8, 59.5],
##            'Fake': [49.1, 41.2, 39.1, 50.5, 52.1],
##            'SAR': [32.2, 33.3, 31.1, 37.4, 34.4]}
##raw_data = {'Images': ['13DEC15', '18MAR16', '05MAY16', '08JUL16', '24JUL16'],
##            'Real': [33.8, 43.6, 45.6, 48.6, 48.3],
##            'Fake': [23.5, 24.2, 29.5, 30.7, 39.3],
##            'SAR': [15.0, 24.8, 21.1, 24.1, 18.8]}
#raw_data = {'Images': ['MAR', 'MAY', 'JUL1', 'JUL2'],
#            'Real': [43.6, 45.6, 48.6, 48.3],
#            'Fake': [24.2, 29.5, 30.7, 39.3],
#            'SAR': [24.8, 21.1, 24.1, 18.8]}
#df = pd.DataFrame(raw_data, columns=['Images', 'Real', 'Fake', 'SAR'])
#print df
#
## Setting the positions and width for the bars
#pos = list(range(len(df['Real'])))
#width = 0.25
#
## Plotting the bars
#fig, ax = plt.subplots(figsize=(10, 5))
#
## Create a bar with pre_score data,
## in position pos,
#plt.bar(pos,
#        # using df['pre_score'] data,
#        df['Real'],
#        # of width
#        width,
#        # with alpha 0.5
#        alpha=0.5,
#        # with color
#        color='#F00000',
#        # with label the first value in first_name
#        label=df['Images'][0],
#        edgecolor='black',
#        )
#
## Create a bar with mid_score data,
## in position pos + some width buffer,
#plt.bar([p + width for p in pos],
#        # using df['mid_score'] data,
#        df['Fake'],
#        # of width
#        width,
#        # with alpha 0.5
#        alpha=0.5,
#        # with color
#        color='#00F000',
#        # with label the second value in first_name
#        label=df['Images'][1],
#        edgecolor='black',
#        hatch="\\")
#
## Create a bar with post_score data,
## in position pos + some width buffer,
#plt.bar([p + width*2 for p in pos],
#        # using df['post_score'] data,
#        df['SAR'],
#        # of width
#        width,
#        # with alpha 0.5
#        alpha=0.5,
#        # with color
#        color='#0000F0',
#        # with label the third value in first_name
#        label=df['Images'][2],
#        edgecolor='black',
#        hatch="/")
#
## Set the y axis label
#ax.set_ylabel('Averge Accuracy (AA)', fontsize='xx-large')
#
## Set the chart's title
#ax.set_title('Monotemporal Classification', fontsize='xx-large')
#
## Set the position of the x ticks
#ax.set_xticks([p + 1.0 * width for p in pos])
#
## Set the labels for the x ticks
#ax.set_xticklabels(df['Images'], fontsize='xx-large')
#ax.set_rasterized(True)
#
## Setting the x-axis and y-axis limits
#plt.xlim(min(pos)-width, max(pos)+width*4)
#plt.ylim([0, 100])
#
## Adding the legend and showing the plot
#plt.legend(['Optical Real', 'Optical Fake', 'SAR'], loc='upper right', fontsize='xx-large')
## plt.grid()
#plt.show()
#fig.tight_layout()
## plt.savefig('/home/jose/Drive/PUC/WorkPlace/ISSPRS/monotemporal_AA4.eps', dpi=100)
#
## #############################################################################
## MULTITEMPORAL
## Overall Acc
##raw_data = {'Images': ['13DEC15', '18MAR16', '05MAY16', '08JUL16', '24JUL16'],
##            'Real': [93.0, 86.2, 91.6, 79.8, 79.8],
##            'Fake': [88.0, 78.3, 87.8, 75.3, 75.3],
##            'SAR': [80.5, 69.9, 81.2, 68.1, 68.1]}
##raw_data = {'Images': ['13DEC15', '18MAR16', '05MAY16', '08JUL16', '24JUL16'],
##            'Real': [92.5, 83.4, 89.5, 73.5, 73.5],
##            'Fake': [86.6, 74.1, 83.7, 68.9, 68.9],
##            'SAR': [77.7, 67.3, 78.7, 63.5, 63.6]}
#raw_data = {'Images': ['MAR', 'MAY', 'JUL1', 'JUL2'],
#            'Real': [81.2, 88.0, 72.4, 72.4],
#            'Fake': [72.3, 82.2, 68.1, 68.2],
#            'SAR': [65.5, 76.1, 61.6, 61.6]}
#df = pd.DataFrame(raw_data, columns=['Images', 'Real', 'Fake', 'SAR'])
#print df
#
## Setting the positions and width for the bars
#pos = list(range(len(df['Real'])))
#width = 0.25
#
## Plotting the bars
#fig, ax = plt.subplots(figsize=(10, 5))
#
## Create a bar with pre_score data,
## in position pos,
#plt.bar(pos,
#        # using df['pre_score'] data,
#        df['Real'],
#        # of width
#        width,
#        # with alpha 0.5
#        alpha=0.5,
#        # with color
#        color='#F00000',
#        # with label the first value in first_name
#        label=df['Images'][0],
#        edgecolor='black',
#        )
#
## Create a bar with mid_score data,
## in position pos + some width buffer,
#plt.bar([p + width for p in pos],
#        # using df['mid_score'] data,
#        df['Fake'],
#        # of width
#        width,
#        # with alpha 0.5
#        alpha=0.5,
#        # with color
#        color='#00F000',
#        # with label the second value in first_name
#        label=df['Images'][1],
#        edgecolor='black',
#        hatch="\\")
#
## Create a bar with post_score data,
## in position pos + some width buffer,
#plt.bar([p + width*2 for p in pos],
#        # using df['post_score'] data,
#        df['SAR'],
#        # of width
#        width,
#        # with alpha 0.5
#        alpha=0.5,
#        # with color
#        color='#0000F0',
#        # with label the third value in first_name
#        label=df['Images'][2],
#        edgecolor='black',
#        hatch="/")
#
## Set the y axis label
#ax.set_ylabel('Accuracy (OA)', fontsize='xx-large')
#
## Set the chart's title
#ax.set_title('Multitemporal Classification', fontsize='xx-large')
#
## Set the position of the x ticks
#ax.set_xticks([p + 1.0 * width for p in pos])
#
## Set the labels for the x ticks
#ax.set_xticklabels(df['Images'], fontsize='xx-large')
#
## Setting the x-axis and y-axis limits
#plt.xlim(min(pos)-width, max(pos)+width*4)
#plt.ylim([0, 100])
#ax.set_rasterized(True)
#
## Adding the legend and showing the plot
#plt.legend(['Optical Real', 'Optical Fake', 'SAR'], loc='upper right', fontsize='xx-large')
## plt.grid()
#plt.show()
#fig.tight_layout()
## plt.savefig('/home/jose/Drive/PUC/WorkPlace/ISSPRS/multitemporal_OA4.eps', dpi=100)
#
## ############################
## Average Acc
##raw_data = {'Images': ['13DEC15', '18MAR16', '05MAY16', '08JUL16', '24JUL16'],
##            'Real': [81.5, 82.7, 76.2, 75.6, 75.6],
##            'Fake': [76.0, 70.7, 62.5, 67.0, 67.0],
##            'SAR': [71.0, 67.9, 66.5, 64.4, 64.4]}
#raw_data = {'Images': ['MAR', 'MAY', 'JUL1', 'JUL2'],
#            'Real': [53.3, 49.2, 56.4, 56.4],
#            'Fake': [46.7, 35.5, 47.1, 47.1],
#            'SAR': [40.6, 31.3, 36.0, 36.0]}
#df = pd.DataFrame(raw_data, columns=['Images', 'Real', 'Fake', 'SAR'])
#print df
#
## Setting the positions and width for the bars
#pos = list(range(len(df['Real'])))
#width = 0.25
#
## Plotting the bars
#fig, ax = plt.subplots(figsize=(10, 5))
#
## Create a bar with pre_score data,
## in position pos,
#plt.bar(pos,
#        # using df['pre_score'] data,
#        df['Real'],
#        # of width
#        width,
#        # with alpha 0.5
#        alpha=0.5,
#        # with color
#        color='#F00000',
#        # with label the first value in first_name
#        label=df['Images'][0],
#        edgecolor='black',
#        )
#
## Create a bar with mid_score data,
## in position pos + some width buffer,
#plt.bar([p + width for p in pos],
#        # using df['mid_score'] data,
#        df['Fake'],
#        # of width
#        width,
#        # with alpha 0.5
#        alpha=0.5,
#        # with color
#        color='#00F000',
#        # with label the second value in first_name
#        label=df['Images'][1],
#        edgecolor='black',
#        hatch="\\")
#
## Create a bar with post_score data,
## in position pos + some width buffer,
#plt.bar([p + width*2 for p in pos],
#        # using df['post_score'] data,
#        df['SAR'],
#        # of width
#        width,
#        # with alpha 0.5
#        alpha=0.5,
#        # with color
#        color='#0000F0',
#        # with label the third value in first_name
#        label=df['Images'][2],
#        edgecolor='black',
#        hatch="/")
#
## Set the y axis label
#ax.set_ylabel('Averge Accuracy (AA)', fontsize='xx-large')
#
## Set the chart's title
#ax.set_title('Multitemporal Classification', fontsize='xx-large')
#
## Set the position of the x ticks
#ax.set_xticks([p + 1.0 * width for p in pos])
#
## Set the labels for the x ticks
#ax.set_xticklabels(df['Images'], fontsize='xx-large')
#ax.set_rasterized(True)
#
## Setting the x-axis and y-axis limits
#plt.xlim(min(pos)-width, max(pos)+width*4)
#plt.ylim([0, 100])
#
## Adding the legend and showing the plot
#plt.legend(['Optical Real', 'Optical Fake', 'SAR'], loc='upper right', fontsize='xx-large')
## plt.grid()
#plt.show()
#fig.tight_layout()
# plt.savefig('/home/jose/Drive/PUC/WorkPlace/ISSPRS/multitemporal_AA4.eps', dpi=100)