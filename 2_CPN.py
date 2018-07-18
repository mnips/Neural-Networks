import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#matplotlib.use('TkAgg')
import seaborn as sns
#sns.set(style='ticks')
from pprint import pprint
import math

'''
Author: Priyansh Soni
'''

### FULL-CPN Algorithm

np.random.seed(0)
input_x=np.matrix([[1,0,0,0,0,0,0,0],
				  [0,1,0,0,0,0,0,0],
				  [0,0,1,0,0,0,0,0],
				  [0,0,0,1,0,0,0,0],
				  [0,0,0,0,1,0,0,0],
				  [0,0,0,0,0,1,0,0],
				  [0,0,0,0,0,0,1,0],
				  [0,0,0,0,0,0,0,1]])
output_y=np.matrix([[0,0,0],
				   [0,0,1],
				   [0,1,0],
				   [0,1,1],
				   [1,0,0],
				   [1,0,1],
				   [1,1,0],
				   [1,1,1]])



#OR Weights can also be initialized as following 1/sqrt(n)
p_cluster_num=50 ##Number of clusters
n_input_num=len(list(input_x[0].flat))  ###8 (Input Nodes - X)
m_output_num=len(list(output_y[0].flat)) ###3 (Output Nodes - Y)

#default_value=1/(math.sqrt(len(input_x)))
##Weights initialization
mat_weights_v=np.random.uniform(low=0.1,high=1,size=(n_input_num,p_cluster_num))
mat_weights_w=np.random.uniform(low=0.1,high=1,size=(m_output_num,p_cluster_num))
mat_weights_u=np.random.uniform(low=0.1,high=1,size=(p_cluster_num,m_output_num))
mat_weights_t=np.random.uniform(low=0.1,high=1,size=(p_cluster_num,n_input_num))
##Other way of weights initialization
# mat_weights_v=np.full((n_input_num,p_cluster_num),default_value)
# mat_weights_w=np.full((m_output_num,p_cluster_num),default_value)
# mat_weights_u=np.full((p_cluster_num,m_output_num),default_value)
# mat_weights_t=np.full((p_cluster_num,n_input_num),default_value)
print("------------------------")
print("Training occurs perfectly with 100% precision as number of clusters are high.")
print("Number of clusters = 50")
print("------------------------")
np.set_printoptions(precision=2,suppress=2)
print("Initial Weight Matrics:")
print("------------------------")
print("V matrix")
print("------------------------")
print(mat_weights_v)
print("------------------------")
print("W matrix")
print("------------------------")
print(mat_weights_w)
print("------------------------")
print("U matrix")
print("------------------------")
print(mat_weights_u)
print("------------------------")
print("T matrix")
print("------------------------")
print(mat_weights_t)
print("------------------------")

##Learning rates
alpha=0.6;a=0.5
beta=0.6; b= 0.5
iterations=0
##Phase 1
while(True):
	for (val,out) in zip(input_x,output_y):
		val=list(val.flat)
		out=list(out.flat)
		D=[]
		
		## Find winning cluster
		for j in range(p_cluster_num):
			sum_input_to_cluster=0
			sum_output_to_cluster=0
			for i in range(n_input_num):
				sum_input_to_cluster+=(val[i]-mat_weights_v[i,j])**2
				
			
			for k in range(m_output_num):
				sum_output_to_cluster+=(out[k]-mat_weights_w[k,j])**2
			
			D.append(sum_input_to_cluster+sum_output_to_cluster)

		winning_cluster_val=min(D)
		
		winning_cluster_index_j=D.index(winning_cluster_val)
		# print(winning_cluster_index_j)
		
		##Update weight from input,output to cluster
		for i in range(n_input_num):
			mat_weights_v[i,winning_cluster_index_j]=(1-alpha)*mat_weights_v[i,winning_cluster_index_j]+alpha*val[i]
		
		for k in range(m_output_num):
			mat_weights_w[k,winning_cluster_index_j]=(1-beta)*mat_weights_w[k,winning_cluster_index_j]+beta*out[k]
		

	#Reduce alpha and beta
	alpha=alpha*0.97
	beta=beta*0.97
	iterations+=1
	##Stopping condition
	if(alpha<=0.001):
		break
##Phase 2
##Alpha and beta small values as obtained earlier kept. We may even retrieve them as commented below.
# alpha=0.6
# beta=0.6
iterations=0
while(True):
	for (val,out) in zip(input_x,output_y):
		val=list(val.flat)
		out=list(out.flat)
		D=[]
		
		## Find winning cluster
		for j in range(p_cluster_num):
			sum_input_to_cluster=0
			sum_output_to_cluster=0
			for i in range(n_input_num):
				sum_input_to_cluster+=(val[i]-mat_weights_v[i,j])**2
				
			
			for k in range(m_output_num):
				sum_output_to_cluster+=(out[k]-mat_weights_w[k,j])**2
			
			D.append(sum_input_to_cluster+sum_output_to_cluster)

		winning_cluster_val=min(D)
		
		winning_cluster_index_j=D.index(winning_cluster_val)
		# print(winning_cluster_index_j)
		
		##Update weight from input,output to cluster
		for i in range(n_input_num):
			mat_weights_v[i,winning_cluster_index_j]=(1-alpha)*mat_weights_v[i,winning_cluster_index_j]+alpha*val[i]
		
		for k in range(m_output_num):
			mat_weights_w[k,winning_cluster_index_j]=(1-beta)*mat_weights_w[k,winning_cluster_index_j]+beta*out[k]
		
		
		##Update weight from cluster to output x*,y*
		for i in range(m_output_num):
			mat_weights_u[winning_cluster_index_j,i]=(1-a)*mat_weights_u[winning_cluster_index_j,i]+a*out[i]
		
		for k in range(n_input_num):
			mat_weights_t[winning_cluster_index_j,k]=(1-b)*mat_weights_t[winning_cluster_index_j,k]+b*val[k]
	iterations+=1
	#Reduce a and b
	# a=a*0.97
	# b=b*0.97
	a=a*0.97
	b=b*0.97
	if(a<=0.001):
		break

np.set_printoptions(precision=2,suppress=2)
print("Final Weight Matrics:")
print("------------------------")
print("V matrix")
print("------------------------")
print(mat_weights_v)
print("------------------------")
print("W matrix")
print("------------------------")
print(mat_weights_w)
print("------------------------")
print("U matrix")
print("------------------------")
print(mat_weights_u)
print("------------------------")
print("T matrix")
print("------------------------")
print(mat_weights_t)
print("------------------------")

plot_input_x=[]
plot_output_y=[]
###Testing 
print("------------------------")
print("Testing")
print("------------------------")
for (x,y) in zip(input_x,output_y):
	x=list(x.flat)
	y=list(y.flat)
	print("------------------------")
	print("Input X and Output Y values:")
	print("X",x)
	print("Y",y)
	print("------------------------")
	D=[]
	##Find j closest to x
	for j in range(p_cluster_num):
		sum_input_to_cluster=0
		sum_output_to_cluster=0
		for i in range(n_input_num):
			sum_input_to_cluster+=(x[i]-mat_weights_v[i,j])**2
				
			
		for k in range(m_output_num):
			sum_output_to_cluster+=(y[k]-mat_weights_w[k,j])**2
			
		D.append(sum_input_to_cluster+sum_output_to_cluster)
	# print("------------------------")
	# print("D_matrix Weights")
	# print("------------------------")
	# print(D)
	# print("------------------------")
	winning_cluster_val=min(D)
	
	winning_cluster_index_j=D.index(winning_cluster_val)
	print("Winning Cluster Number",winning_cluster_index_j)
	print("------------------------")
	
	print("Final X-star values:")
	print(mat_weights_t[winning_cluster_index_j,:])
	print("Final Y-star values:")
	print(mat_weights_u[winning_cluster_index_j,:])
	print("------------------------")
