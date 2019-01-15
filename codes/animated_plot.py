import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 6*np.pi, 100)
y = np.sin(x)

# You probably won't need this if you're embedding things in a tkinter plot...
#plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma
#
#cont = 0.0
#for phase in np.linspace(0, 10*np.pi, 500):
##    line1, = ax.plot(x + phase, y, 'r-')
#    cont +=1
#    print(phase)
#    plt.xlim(0, (x[-1] + cont/len(x)))
##    plt.xlim(0, phase)
#    line1.set_data(x[-1] + cont/len(x), np.sin(x[-1] + phase))
#    plt.pause(0.1   )
##    fig.canvas.draw()
##    fig.canvas.flush_events()
