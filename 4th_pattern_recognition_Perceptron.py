import numpy as np
import pandas as pd
import math
import copy
import matplotlib
import operator
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

'''
Author:Priyansh Soni
'''

##PATTERN CLASSIFICATION

def Testing(mat_pattern,mat_inputs_s,mat_weight,bias):
	print("------------------------")
	print("Testing for PATTERN RECOGNITION")
	print("------------------------")
	for i in range(len(mat_inputs_s)):
		mat_inputs_x=list(mat_inputs_s[i])
		val=bias+sum([mat_weight[i]*mat_inputs_x[i] for i in range(len(mat_inputs_x))])
		if(val<=0): 
			output=-1
		else:
			output=1
		if(output==1):
			output='T'
		else:
			output='H'
		print("Input: '#' represents 1 and '*' represents -1")
		print(np.reshape(mat_pattern[i], (3, 3)))
		
		print("Output: "+str(output)+" (Value="+str(val)+")" )

def PerceptronModel(alpha,theta):
	##Input and Target
	mat_inputs_s=[['#','#','#','*','#','*','*','#','*'],['#','*','#','#','#','#','#','*','#']]
	mat_original_inputs=copy.deepcopy(mat_inputs_s)
	
	##Noisy inputs
	mat_mistake_inputs=[['*','*','#','*','#','*','*','#','*'],['*','#','#','#','#','#','#','*','#']]
	mat_original_mistake_inputs=copy.deepcopy(mat_mistake_inputs)
	##Missing inputs
	mat_missing_inputs=[[0,0,'#','*','#','*','*','#','*'],[0,0,'#','#','#','#','#','*','#']]
	mat_original_missing_inputs=copy.deepcopy(mat_missing_inputs)
	##Convert to bipolar input
	for i in range(len(mat_inputs_s)):
		mat_inputs_s[i]=[1 if x=='#' else -1 for x in mat_inputs_s[i]]
		mat_mistake_inputs[i]=[1 if x=='#' else -1 for x in mat_mistake_inputs[i]]
		mat_missing_inputs[i]=[1 if x=='#' else -1 for x in mat_missing_inputs[i]]

	#print(mat_inputs_s)

	mat_target_t=['T','H']
	mat_bipolar_target_t=[1,-1] # 1 for T and 0 for H
	#mat_target_t=mat_bipolar_target_t
	##Inititally weights and bias 0 (can be taken random as well)
	b=iterations=0 
	mat_weight=[0]*len(mat_inputs_s[0])
	
	curr_weight=mat_weight
	print("------------------------")
	print("Initial Weights:", mat_weight)
	print("------------------------")

	while(True):
		iterations+=1
		prev_weight=curr_weight

		for i in range(len(mat_inputs_s)):
			mat_inputs_x_i=list(mat_inputs_s[i])
			
			##Computation of response unit (y_in)
			y_in=b+sum([mat_inputs_x_i[j]*mat_weight[j] for j in range(len(mat_inputs_x_i))])
			#print("Y_in:",y_in)		
			
			if(y_in>theta):
				y=1
			elif(y_in<-1*theta):
				y=-1
			else:
				y=0
			##If error occurred means y!=target or t
			if(y!=mat_bipolar_target_t[i]):##w1 and w2 change both
				mat_weight=[mat_weight[j]+alpha*mat_bipolar_target_t[i]*mat_inputs_x_i[j] for j in range(len(mat_weight)) ]
				b=b+alpha*mat_bipolar_target_t[i]
			print("------------------------")
			print("Updated weights and bias:")
			print("Iteration: "+str(iterations)+", Pattern: "+str(mat_target_t[i]))
			print("Weights:",mat_weight)
			print("Bias:",b)
			print("------------------------")
			
		curr_weight=mat_weight
		
		##Stopping Condition
		if(prev_weight==curr_weight):
			print("Stopping condition occured. Number of Iterations for convergence:",iterations)
			print("Final Weights:",mat_weight)
			print("Bias:",b)
			print("------------------------")
			break
	print("------------------------")		
	print("Testing for original pattern")	
	Testing(mat_original_inputs,mat_inputs_s,mat_weight,b)
	print("------------------------")	
	print("Testing for mistake pattern(2 wrong values)")	
	Testing(mat_original_mistake_inputs,mat_mistake_inputs,mat_weight,b)
	print("------------------------")	
	print("Testing for missing pattern(2 missing values)")	
	Testing(mat_original_missing_inputs,mat_missing_inputs,mat_weight,b)


PerceptronModel(1,0.1) #alpha=1,theta=0.1