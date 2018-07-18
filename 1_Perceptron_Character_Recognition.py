import numpy as np
import copy
import pandas as pd
import math
import matplotlib
import operator
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
'''
Author:Priyansh Soni
'''

def Testing(mat_inputs_s,mat_weight,bias):
	print("------------------------")
	print("Testing for PATTERN RECOGNITION")
	print("------------------------")
	characters_list=['C','D','E','I','T']
	for i in range(len(mat_inputs_s)):
		mat_inputs_x=list(mat_inputs_s[i])
		output=sum([mat_weight[i]*mat_inputs_x[i] for i in range(len(mat_inputs_x))])
		Activated_final_output=[]
		for val in output:
			if(val<=0): 
				Activated_final_output.append(-1)
			else:
				Activated_final_output.append(1)
		print("------------------------")
		print("Input Pattern")
		Display(mat_inputs_x)

		print("Output: "+str(Activated_final_output))
		for i, j in enumerate(Activated_final_output):
			if j==1:
				print("Alphabet Recognized as: "+characters_list[i])


def Display(Input_Pattern):
	Display_Pattern=np.reshape(['#' if x==1 else '.' if x==-1 else '0' for x in Input_Pattern],(9,7))
	
	for row in Display_Pattern:
		print (" ".join(map(str,row)))
	print()
def Activation_function(input,theta):
	
	if(input>=theta):
		return 1
	elif(input<-1*theta):
		return -1
	else:
		return 0
	
def InputPatterns():
	C1=[-1,-1,1,1,1,1,1,
	-1,1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,-1,
	-1,1,-1,-1,-1,-1,-1,
	-1,-1,1,1,1,1,1]
	C2=[-1,-1,1,1,1,-1,-1,
	-1,1,-1,-1,-1,1,-1,
	1,-1,-1,-1,-1,-1,1,
	1,-1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,1,
	-1,1,-1,-1,-1,1,-1,
	-1,-1,1,1,1,-1,-1]
	C3=[1,1,1,1,1,1,1,
	1,-1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,-1,
	1,1,1,1,1,1,1]

	print("--------------")
	print("   Font 1")
	print("--------------")
	Display(C1)
	print("--------------")
	print("   Font 2")
	print("--------------")
	Display(C2)
	print("--------------")
	print("   Font 3")
	print("--------------")
	Display(C3)

	D1=[1,1,1,1,1,-1,-1,
	-1,1,-1,-1,-1,1,-1,
	-1,1,-1,-1,-1,-1,1,
	-1,1,-1,-1,-1,-1,1,
	-1,1,-1,-1,-1,-1,1,
	-1,1,-1,-1,-1,-1,1,
	-1,1,-1,-1,-1,-1,1,
	-1,1,-1,-1,-1,1,-1,
	1,1,1,1,1,-1,-1]
	
	D2=[1,1,1,1,1,-1,-1,
	1,-1,-1,-1,-1,1,-1,
	1,-1,-1,-1,-1,-1,1,
	1,-1,-1,-1,-1,-1,1,
	1,-1,-1,-1,-1,-1,1,
	1,-1,-1,-1,-1,-1,1,
	1,-1,-1,-1,-1,-1,1,
	1,-1,-1,-1,-1,1,-1,
	1,1,1,1,1,-1,-1]
	
	D3=[1,1,1,1,1,1,-1,
	-1,1,-1,-1,-1,-1,1,
	-1,1,-1,-1,-1,-1,1,
	-1,1,-1,-1,-1,-1,1,
	-1,1,-1,-1,-1,-1,1,
	-1,1,-1,-1,-1,-1,1,
	-1,1,-1,-1,-1,-1,1,
	-1,1,-1,-1,-1,-1,1,
	1,1,1,1,1,1,-1]

	Display(D1)
	Display(D2)
	Display(D3)

	I1=[-1,-1,1,1,1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,1,1,1,-1,-1]
	I2=[1,1,1,1,1,1,1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	1,1,1,1,1,1,1]
	I3=[-1,1,1,1,1,1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,1,1,1,1,1,-1]
	Display(I1)
	Display(I2)
	Display(I3)

	E1=[1,1,1,1,1,1,1,
	-1,1,-1,-1,-1,-1,1,
	-1,1,-1,-1,-1,-1,-1,
	-1,1,-1,1,-1,-1,-1,
	-1,1,1,1,-1,-1,-1,
	-1,1,-1,1,-1,-1,-1,
	-1,1,-1,-1,-1,-1,-1,
	-1,1,-1,-1,-1,-1,1,
	1,1,1,1,1,1,1]
	E2=[1,1,1,1,1,1,1,
	1,-1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,-1,
	1,1,1,1,1,1,1,
	1,-1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,-1,
	1,1,1,1,1,1,1]
	E3=[1,1,1,1,1,1,1,
	-1,1,-1,-1,-1,-1,-1,
	-1,1,-1,-1,-1,-1,-1,
	-1,1,-1,-1,-1,-1,-1,
	-1,1,1,1,1,1,-1,
	-1,1,-1,-1,-1,-1,-1,
	-1,1,-1,-1,-1,-1,-1,
	-1,1,-1,-1,-1,-1,-1,
	1,1,1,1,1,1,1]
	Display(E1)
	Display(E2)
	Display(E3)

	T1=[1,1,1,1,1,1,1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1]
	T2=[1,1,1,1,1,1,1,
	1,-1,-1,1,-1,-1,1,
	1,-1,-1,1,-1,-1,1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1]
	T3=[1,-1,-1,-1,-1,-1,1,
	1,1,1,1,1,1,1,
	1,-1,-1,1,-1,-1,1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1]
	Display(T1)
	Display(T2)
	Display(T3)
	output_C=[1,-1,-1,-1,-1]
	output_D=[-1,1,-1,-1,-1]
	output_E=[-1,-1,1,-1,-1]
	output_I=[-1,-1,-1,1,-1]
	output_T=[-1,-1,-1,-1,1]

	training_patterns=[C1,C2,C3,D1,D2,D3,E1,E2,E3,I1,I2,I3,T1,T2,T3]
	target_patterns=[output_C,output_C,output_C,output_D,output_D,output_D,output_E,output_E,output_E,output_I,output_I,output_I,output_T,output_T,output_T]
	return training_patterns,target_patterns

def NoisyTestingData():
	### 3 mistakes and 2 missing
	## Mistakes are -1 replaced by 1 and vice versa and missing are where 0 introduced
	
	Noise_C=[1,-1,1,1,1,-1,-1,
	1,1,-1,-1,-1,1,-1,
	-1,-1,-1,-1,-1,-1,1,
	0,-1,-1,-1,-1,-1,-1,
	0,-1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,1,
	-1,1,-1,-1,-1,1,-1,
	-1,-1,1,1,1,-1,-1]
	Noise_D=[-1,1,1,1,1,-1,-1,
	-1,-1,-1,-1,-1,1,-1,
	-1,-1,-1,-1,-1,-1,1,
	0,-1,-1,-1,-1,-1,1,
	0,-1,-1,-1,-1,-1,1,
	1,-1,-1,-1,-1,-1,1,
	1,-1,-1,-1,-1,-1,1,
	1,-1,-1,-1,-1,1,-1,
	1,1,1,1,1,-1,-1]
	Noise_I=[-1,1,1,1,1,1,1,
	1,-1,-1,1,-1,-1,-1,
	1,-1,-1,1,-1,-1,-1,
	0,-1,-1,1,-1,-1,-1,
	0,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	1,1,1,1,1,1,1]
	Noise_E=[-1,1,1,1,1,1,1,
	-1,-1,-1,-1,-1,-1,-1,
	-1,-1,-1,-1,-1,-1,-1,
	0,-1,-1,-1,-1,-1,-1,
	0,1,1,1,1,1,1,
	1,-1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,-1,
	1,-1,-1,-1,-1,-1,-1,
	1,1,1,1,1,1,1]
	Noise_T=[-1,1,1,1,1,1,1,
	-1,-1,-1,1,-1,-1,1,
	-1,-1,-1,1,-1,-1,1,
	0,-1,-1,1,-1,-1,-1,
	0,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1,
	-1,-1,-1,1,-1,-1,-1]
	NoisyPatterns=[Noise_C,Noise_D,Noise_E,Noise_I,Noise_T]

	return NoisyPatterns

def PerceptronModel_CharacterRecognition(alpha,theta,mat_inputs_s,mat_target_t,noisy_inputs):
	#Weight matrix for 9*7=63 inputs
	mat_weight=[0]*63*5 ##Weight matrix 63(inputs)* 5 (outputs/classes)
	mat_weight=np.reshape(mat_weight,(63,5))

	b=[0]*5 ## 5 classes of output

	iterations=0
	while(True):
		prev_weight=copy.deepcopy(mat_weight)
		n=63 ##No. of input units (63=9*7)-input pattern length and width
		m=5 ##5 output units
		for pair_num in range(len(mat_inputs_s)):
			## Select first input pattern out of 15.
			mat_inputs_x_i=mat_inputs_s[pair_num]
			
			for j in range(m):
				##Computation of y_in for each output unit
				y_in_j=b[j]+np.dot(mat_inputs_x_i,mat_weight[:,j])
				##Activation for each output unit
				y_j=Activation_function(y_in_j,theta)
				##This can also be done using error
				# error=mat_target_t[pair_num][j]-y_j
				# error=alpha*error
				##In case error occured and target and y_j do not match
				if(mat_target_t[pair_num][j]!=y_j):
					##Updation of weights and bias
					b[j]=b[j]+alpha*mat_target_t[pair_num][j]
					for k in range(63):
						mat_weight[k,j]+=alpha*mat_target_t[pair_num][j]*mat_inputs_x_i[k]
					##This can be used in case error is used.
					#mat_weight[:,j]=mat_weight[:,j]+np.dot(mat_inputs_x_i,np.array(error))
					
		iterations=iterations+1
		#print("Updated weights:");print(np.matrix(mat_weight))
		if((prev_weight==mat_weight).all()):
			print("------------------------")
			print("Stopping condition occured. Number of Iterations for convergence:",iterations)
			print("Bias:",b)
			print("Final Weights:");
			print(mat_weight)
			print("------------------------")
			
			#print(np.dot(mat_inputs_s[3],mat_weight))
			break
	print("------------------------")
	print("Testing DEFAULT INPUTS")
	Testing(mat_inputs_s,mat_weight,b)
	print("------------------------")
	print("Testing NOISY INPUTS")
	Testing(noisy_inputs,mat_weight,b)
	print("------------------------")

### Getting Input and Noisy Patterns and Running Model(alpha=1, theta=0.1)
input_pattern_list,target_pattern_list=InputPatterns()
noisy_pattern_list=NoisyTestingData()

PerceptronModel_CharacterRecognition(1,0.1,input_pattern_list,target_pattern_list,noisy_pattern_list) 

