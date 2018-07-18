import pandas as pd
import numpy as np
from pprint import pprint
import math
import copy

'''
Author: Priyansh Soni
'''

def ART1_Algorithm(Rho):
	##ART1 Algorithm
	n=7; # Input units = 7
	m=1; # Cluster units = 1

	L=(n-1)/n ## so that bij=L/(L-1+n) = 1/(n+1) ; L is also used alternatively with alpha

	mat_input=np.array([[1, 1, 0, 0, 0, 0, 1],
						[0, 0, 1, 1, 1, 1, 0],
						[1, 0, 1, 1, 1, 1, 0],
						[0, 0, 0, 1, 1, 1, 0],
						[1, 1, 0, 1, 1, 1, 0]])

	t_weights=np.asarray(np.full((n,m),1))#Top down weights
	b_weights=np.asarray(np.full((m,n),L/(L-1+n)))#Bottom up weights
	print("Initial Top-Down Weights")
	print("------------------------")
	print(t_weights)
	print("------------------------")
	print("Initial Bottom-Up Weights")
	print("------------------------")
	print(b_weights)
	print("------------------------")

	iterations=0
	while(True):

		iterations=iterations+1
		print("Iteration: ",iterations)
		print("------------------------")
		old_t_weights =(t_weights)
		old_b_weights =b_weights

		for x_in in mat_input:
			x_input=list(x_in.flat)
			print("For input pattern",x_input)
			print("------------------------")
			activation_F2_layer_y = [0] * m
			for j in range(m):
				node = activation_F2_layer_y[j]
				##Node in F2 layer not inhibited
				if (node != -1):
					## Compute activation for each node in F2 or Cluster layer
					activation_F2_layer_y[j] = sum([b_weights[j][i] * x_input[i] for i in range(n)])
				#print("Cluster node",j+1)
				#print(activation_F2_layer_y[j])
				#print("------------------")

			while(True):
				max_J=max(activation_F2_layer_y)
				index_J=activation_F2_layer_y.index(max_J)
				s_input=[t_weights[i][index_J]*x_input[i] for i in range(n)]
				ratio_similarity=sum(s_input)/sum(x_input)


				if(ratio_similarity<=Rho):
					activation_F2_layer_y[index_J]=-1
				else:
					print("Winning Cluster is", index_J+1)

					x_norm=sum([t_weights[i][index_J]*x_input[i] for i in range(n)])
					for i in range(n):
						b_weights[index_J][i]=(t_weights[i][index_J]*x_input[i])/(0.5+x_norm)
						t_weights[i][index_J]=t_weights[i][index_J]*x_input[i]
						
					break

				if(max_J==-1):
					break

			temp=max(activation_F2_layer_y)
			##Condition for adding new node
			if(temp==-1):
				top_new_weights=x_input
				bottom_new_weights=[0]*n
				for i in range(n):
					bottom_new_weights[i]=x_input[i]/(0.5+sum(x_input))

				top_new_weights=np.asarray(top_new_weights)
				bottom_new_weights=np.asarray(bottom_new_weights)

				b_weights = np.vstack([b_weights, bottom_new_weights])
				t_weights = np.hstack([t_weights, top_new_weights[:,None]])
				print("New Cluster Added and Total Clusters =", m+1) 
				# print("New Top-Down Weights")
				# print("------------------------")
				# print(t_weights)
				# print("------------------------")
				# print("New Bottom-Up Weights")
				# print("------------------------")
				# print(b_weights)
				# print("------------------------")

				m=m+1
			np.set_printoptions(precision=3)
			
			print("------------------------")
			print("After all weight Updations:")
			print("New Top-Down Weights")
			print("------------------------")
			print(t_weights)
			print("------------------------")
			print("New Bottom-Up Weights")
			print("------------------------")
			print(b_weights)					
			print("------------------------")

		##Stopping condition that weights stop changing
		if(np.array_equal(old_t_weights,t_weights) and np.array_equal(old_b_weights,b_weights)):
			print("Total Iterations:",iterations)
			print("Total clusters formed for Vigilance parameter = "+str(Rho)+" are: "+str(m)+" clusters")
			print("------------------------")
			break



##Vigilance parameter=0.2,0.6,0.9
print("--------------------------")
print("Vigilance parameter = 0.2")
print("--------------------------")
ART1_Algorithm(Rho=0.2)
print("--------------------------")
print("Vigilance parameter = 0.6")
print("--------------------------")
ART1_Algorithm(Rho=0.6)
print("--------------------------")
print("Vigilance parameter = 0.9")
print("--------------------------")
ART1_Algorithm(Rho=0.9)





