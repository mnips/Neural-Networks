import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.patches as mpatches
#matplotlib.use('TkAgg')
import seaborn as sns
#sns.set(style='ticks')

import matplotlib.pyplot as plt

matplotlib.pyplot.style.use('ggplot')

'''
Author:Priyansh Soni
'''

def bipolar_sigmoid(x,sigma):
    a = []
    for item in x:
        a.append((1-math.exp(-sigma*item))/(1+math.exp(-sigma*item)))
    return a
def logistic_sigmoid(x,alpha):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-alpha*item)))
    return a
def graph_basic():
	plt.xlim(-5,5,1)
	plt.ylim(-1.2,1.2,0.5)
	plt.xticks(np.arange(-5, 6, 1))
	plt.yticks(np.arange(-1,2,1))
	###X and Y axis plot
	plt.axhline(0, color='black')
	plt.axvline(0, color='black')


x=list(np.arange(-5.0,6.0,0.0001))
threshold=0

###STEP FUNCTION 
y= [1 if i>threshold else 0 for i in x]
plt.subplot(3,3,1)
graph_basic()
plt.title('Step Function')
plt.step(x,y,label='step function',color='b')

###SIGNUM FUNCTION
y=[1 if i>threshold else -1 if i<threshold else 0 for i in x]
plt.subplot(3,3,2)
graph_basic()
plt.title('Signum Function')
plt.plot(x,y,label='signum function',color='r')
###LINEAR FUNCTION
plt.subplot(3,3,3)
graph_basic()
plt.title('Linear Function')
plt.plot(x,x,label='linear function',color='m')
###RELU FUNCTION
y=[i if i>threshold else 0 for i in x]
plt.subplot(3,3,4)
graph_basic()
plt.title('RELU Function')
plt.plot(x,y,label='RELU function',color='g')

###LOGISTIC SIGMOID
plt.subplot(3,3,5)
graph_basic()
plt.title('Logistic Sigmoid')
logistic_sig = logistic_sigmoid(x,1)
plt.plot(x,logistic_sig,label='logistic sigmoid',color='y')
logistic_sig = logistic_sigmoid(x,2)
plt.plot(x,logistic_sig,label='logistic sigmoid',color='b')
logistic_sig = logistic_sigmoid(x,3)
plt.plot(x,logistic_sig,label='logistic sigmoid',color='m')
yellow_patch = mpatches.Patch(color='yellow', label='Alpha=1')
blue_patch = mpatches.Patch(color='blue', label='Alpha=2')
magenta_patch = mpatches.Patch(color='m', label='Alpha=3')
plt.legend(handles=[yellow_patch, blue_patch,magenta_patch],prop={'size': 6})

###BIPOLAR SIGMOID
plt.subplot(3,3,6)
graph_basic()
plt.title('Bipolar Sigmoid')
bipolar_sig = bipolar_sigmoid(x,1)
plt.plot(x,bipolar_sig,label='bipolar sigmoid',color='g')
bipolar_sig = bipolar_sigmoid(x,2)
plt.plot(x,bipolar_sig,label='bipolar sigmoid',color='m')
bipolar_sig = bipolar_sigmoid(x,3)
plt.plot(x,bipolar_sig,label='bipolar sigmoid',color='b')
green_patch = mpatches.Patch(color='g', label='Alpha=1')
magenta_patch = mpatches.Patch(color='m', label='Alpha=2')
blue_patch = mpatches.Patch(color='b', label='Alpha=3')
plt.legend(handles=[green_patch,magenta_patch,blue_patch],prop={'size': 6})

###HYPERBOLIC TANGENT
bipolar_sig = bipolar_sigmoid(x,2)
plt.subplot(3,3,8)
graph_basic()
plt.title('Hyperbolic Tangent')
plt.plot(x,bipolar_sig,label='hyperbolic tangent',color='c')
plt.tight_layout()

plt.show()
plt.close()