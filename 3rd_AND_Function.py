import numpy as np
import pandas as pd
import math
import matplotlib
import operator
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
'''
Author:Priyansh Soni
'''

##AND GATE 
def Testing(mat_inputs_s,mat_weight,bias):
	print("------------------------")
	print("Testing for AND GATE")
	print("------------------------")
	for i in range(len(mat_inputs_s)):
		mat_inputs_x_i=list(mat_inputs_s[i].flat)
		val=mat_weight[0]*mat_inputs_x_i[0]+mat_weight[1]*mat_inputs_x_i[1]+bias
		if(val<=0): 
			output=-1
		else:
			output=1
		print("Input: "+ str(list(mat_inputs_s[i].flat))+ " Output: "+str(output)+" (Value="+str(val)+")" )
def HebbNet():
	##Inputs for AND GATE
	print("------------------------")
	print("HEBBNET")
	print("------------------------")
	mat_inputs_s=np.matrix([[1,1],[1,-1],[-1,1],[-1,-1]])
	##Target Vector corresponding to Inputs
	mat_target_t=[1,-1,-1,-1]
	w1=w2=b=0 ##Inititally weights and bias 0 according to algorithm
	mat_weight=[w1,w2]
	#print(len(mat_inputs_s))
	for i in range(len(mat_inputs_s)):
		mat_inputs_x_i=list(mat_inputs_s[i].flat)
		#print(mat_inputs_x_i)
		output_y=(mat_target_t[i])
		#print(output_y)
		#Weight and bias updation
		for j in range(len(mat_weight)):
			mat_weight[j]=mat_weight[j]+mat_inputs_x_i[j]*output_y
		b=b+output_y
		print("Weights after Pattern "+str(i+1)+": "+ str(mat_weight))
		print("Bias:",b)
	print("------------------------")
	print("Final Weights:", mat_weight)
	print("Final Bias:",b)
	print("Equation making the problem Linearly separable:")
	print(str(mat_weight[0])+"*x1 + "+str(mat_weight[0])+"*x2 "+str(b)+" = 0")
	###TESTING
	# for i in range(len(mat_inputs_s)):
	# 	mat_inputs_x_i=list(mat_inputs_s[i].flat)
	# 	val=mat_weight[0]*mat_inputs_x_i[0]+mat_weight[1]*mat_inputs_x_i[1]+b
	# 	if(val<=0): 
	# 		output=-1
	# 	else:
	# 		output=1
	# 	print("Input: "+ str(list(mat_inputs_s[i].flat))+ " Output: "+str(output)+" (Value="+str(val)+")" )
	Testing(mat_inputs_s,mat_weight,b)

def PerceptronRule(alpha,theta):
	print("------------------------")
	print("PERCEPTRON RULE")
	print("------------------------")
	##Inputs for AND GATE (alpha param ranges from 0 to 1 (0 1], 0 excluded
	mat_inputs_s=np.matrix([[1,1],[1,-1],[-1,1],[-1,-1]])
	##Target Vector corresponding to Inputs
	mat_target_t=[1,-1,-1,-1]
	w1=w2=b=iterations=0 ##Inititally weights and bias 0 (can be taken random as well)
	mat_weight=[w1,w2]
	prev_weight=[100,100]
	curr_weight=mat_weight
	print("------------------------")
	print("Initial Weights:", mat_weight)
	print("------------------------")
	while(True):
		iterations+=1
		prev_weight=curr_weight

		for i in range(len(mat_inputs_s)):
			mat_inputs_x_i=list(mat_inputs_s[i].flat)
			##Computation of response unit (y_in)
			y_in=b+sum([mat_inputs_x_i[j]*mat_weight[j] for j in range(len(mat_inputs_x_i))])
			#print("Y_in:",y_in)		
			#print(sum([mat_inputs_x_i[j]*mat_weight[j] for j in range(len(mat_inputs_x_i))]))
			if(y_in>theta):
				y=1
			elif(y_in<-1*theta):
				y=-1
			else:
				y=0
			##If error occurred means y!=target or t
			if(y!=mat_target_t[i]):##w1 and w2 change both
				mat_weight=[mat_weight[j]+alpha*mat_target_t[i]*mat_inputs_x_i[j] for j in range(len(mat_weight)) ]
				b=b+alpha*mat_target_t[i]
			print("------------------------")
			print("Updated weights and bias:")
			print("Iteration: "+str(iterations)+", Pattern: "+str(i+1))
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
	Testing(mat_inputs_s,mat_weight,b)
		
def DeltaRule(alpha):
	print("------------------------")
	print("DELTA RULE OF LEARNING")
	print("------------------------")
	##Inputs for AND GATE (alpha param ranges from 0 to 1 (0 1], 0 excluded
	mat_inputs_s=np.matrix([[1,1],[1,-1],[-1,1],[-1,-1]])
	##Target Vector corresponding to Inputs
	mat_target_t=[1,-1,-1,-1]
	w1=w2=b=iterations=0 ##Inititally weights and bias to be taken randomly
	mat_weight=[w1,w2]
	print("------------------------")
	print("Initial Weights:", mat_weight)
	print("------------------------")
	##Error threshold set
	error_threshold=0.001

	while(True):
		iterations+=1
		largest_weight_change=-1 ##initially a low number
		curr_weight=mat_weight
		for i in range(len(mat_inputs_s)):
			mat_inputs_x_i=list(mat_inputs_s[i].flat)
			##Computation of response unit (y_in)
			y_in=b+sum([mat_inputs_x_i[j]*mat_weight[j] for j in range(len(mat_inputs_x_i))])
			#print("Y_in=",y_in)	
			##NO activation function here as activation function=identity function (y=y_in)
			##Weight and bias updation
			b=b+alpha*(mat_target_t[i]-y_in)
			mat_weight=[mat_weight[j]+alpha*(mat_target_t[i]-y_in)*mat_inputs_x_i[j] for j in range(len(mat_weight)) ]
		
		print("------------------------")
		print("Updated weights and bias:")
		print("Iteration: ",iterations)
		print("Weights:",mat_weight)
		print("Bias:",b)
		print("------------------------")	

		##Updating largest weight change
		largest_weight_change=max(map(abs,list(map(operator.sub, mat_weight, curr_weight))))
		##Stopping condition
		if(largest_weight_change<error_threshold):
			print("Stopping condition occured. Number of Iterations:",iterations)
			print("Final Weights:",mat_weight)
			print("Bias:",b)
			print("------------------------")
			break

	Testing(mat_inputs_s,mat_weight,b)

HebbNet()
PerceptronRule(1,0.1) #alpha=1,theta=0.1
DeltaRule(0.1) # 0.1<=n*alpha <= 1.0 (0.025<=alpha<=0.25).  Set alpha=0.1

