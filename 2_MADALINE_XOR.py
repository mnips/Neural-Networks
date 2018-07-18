import numpy as np
import copy
import pandas as pd
import math
import matplotlib
import operator
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

'''
Author:Priyansh Soni
'''

def Activation_function(val):
	if val>=0:
		return 1
	else:
		return -1

def Testing(mat_inputs_s,w11,w21,w12,w22,b1,b2,v1,v2,b3):
	print("------------------------")
	print("Testing for XOR GATE")
	print("------------------------")
	for i in range(len(mat_inputs_s)):
		mat_inputs_x_i=list(mat_inputs_s[i].flat)
		z_in_1=b1+mat_inputs_x_i[0]*w11+mat_inputs_x_i[1]*w21
		z_in_2=b2+mat_inputs_x_i[0]*w12+mat_inputs_x_i[1]*w22
		z1=Activation_function(z_in_1)
		z2=Activation_function(z_in_2)
		y_in=b3+z1*v1+z2*v2
		y=Activation_function(y_in)
		print("Input: "+ str(mat_inputs_x_i)+ " Output: "+str(y)+" (Value="+str(y_in)+")" )


##XOR GATE => (2*AND NOT + OR)
def Madaline_DeltaRule(alpha):
	##Inputs for XOR GATE
	print("------------------------")
	print("MADALINE FOR XOR GATE with alpha ="+str(alpha))
	print("------------------------")
	mat_inputs_s=np.matrix([[1,1],[1,-1],[-1,1],[-1,-1]])
	mat_target_t=[-1,1,1,-1]
	#Fixed Weights and bias for last layer
	v1=v2=b3=0.5
	#Variable weights and bias
	#Weights for Z1
	w11=w21=b1=0
	#Weights for Z2
	w12=w22=b2=0
	iterations=0
	w11=0.05
	w21=0.2
	w12=0.1
	w22=0.2
	b1=0.3
	b2=0.15

	while(True):
		iterations+=1
		prev_w11=copy.deepcopy(w11);prev_w21=copy.deepcopy(w21);prev_w12=copy.deepcopy(w12);prev_w22=copy.deepcopy(w22)
		#prev_w11=(w11);prev_w21=(w21);prev_w12=(w12);prev_w22=(w22)
		for i in range(len(mat_inputs_s)):
			mat_inputs_x_i=list(mat_inputs_s[i].flat)
			#Net Input to hidden ADALINE UNIT
			z_in_1=b1+mat_inputs_x_i[0]*w11+mat_inputs_x_i[1]*w21
			z_in_2=b2+mat_inputs_x_i[0]*w12+mat_inputs_x_i[1]*w22
			#Output enumeration
			z1=Activation_function(z_in_1)
			z2=Activation_function(z_in_2)
			#Output of net (Final layer)
			y_in=b3+z1*v1+z2*v2
			y=Activation_function(y_in)
			##Error
			if(mat_target_t[i]!=y):
				if(mat_target_t[i]==1):
					#Choose node z_input closest to 0 (as both negative so less negative will become 1 faster)
					if(z_in_1>z_in_2):
						#Update Adaline z1
						#print("Target=1 but is -1")
						#print("Z_in_1:",z_in_1)
						#print("Z_in_2:",z_in_2)
						b1=	b1+ alpha*(1-z_in_1)
						w11=w11+alpha*(1-z_in_1)*mat_inputs_x_i[0]
						w21=w21+alpha*(1-z_in_1)*mat_inputs_x_i[1]
					else:
						#Update Adaline z2
						#Update Adaline z1
						#print("Target=1 but is -1")
						#print("Z_in_1:",z_in_1)
						#print("Z_in_2:",z_in_2)
						b2=	b2+ alpha*(1-z_in_2)
						w12=w12+alpha*(1-z_in_2)*mat_inputs_x_i[0]
						w22=w22+alpha*(1-z_in_2)*mat_inputs_x_i[1]
				else: ##Target=-1
					#Update those z_input that are positive (i.e z_input>0)
					if(z_in_1>=0):
						#Update Adaline z1
						b1=	b1+ alpha*(-1-z_in_1)
						w11=w11+alpha*(-1-z_in_1)*mat_inputs_x_i[0]
						w21=w21+alpha*(-1-z_in_1)*mat_inputs_x_i[1]
					if(z_in_2>=0):
						#Update Adaline z2
						b2=	b2+ alpha*(-1-z_in_2)
						w12=w12+alpha*(-1-z_in_2)*mat_inputs_x_i[0]
						w22=w22+alpha*(-1-z_in_2)*mat_inputs_x_i[1]


		##Weight change comparison (Stopping Condition)
		if(prev_w11==w11 and prev_w21==w21 and prev_w12==w12 and prev_w22==w22):
			print("------------------------")
			print("Stopping Condition satisfied. Weights stopped changing.")
			print("Total Iterations = "+str(iterations))
			print("------------------------")
			print("Final Weights:")
			print("------------------------")
			print("Adaline Z1:")
			print("w11 = "+str(w11))
			print("w21 = "+str(w21))
			print("b1 = "+str(b1))
			print("------------------------")
			print("Adaline Z2:")
			print("w12 = "+str(w12))
			print("w22 = "+str(w22))
			print("b2 = "+str(b2))
			print("------------------------")
			Testing(mat_inputs_s,w11,w21,w12,w22,b1,b2,v1,v2,b3)
			break


		print("Iteration = " +str(iterations))
		print("------------------------")
		print("Weights till now:")
		print("------------------------")
		print("Adaline Z1:")
		print("w11 = "+ str(w11) )
		print("w21 = "+ str(w21))
		print("b1 = "+str(b1))
		print("------------------------")
		print("Adaline Z2:")
		print("w12 = "+ str(w12))
		print("w22 = "+ str(w22))
		print("b2 = "+ str(b2))
		print("------------------------")

##Alpha=0.05
Madaline_DeltaRule(0.05)
##Alpha=0.1
Madaline_DeltaRule(0.1)
##Alpha=0.5
Madaline_DeltaRule(0.5)



