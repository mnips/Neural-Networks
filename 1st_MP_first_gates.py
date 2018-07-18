import numpy as np
import pandas as pd

'''
Author: Priyansh Soni

y=(W^T).X , (Weight)^Transpose.Inputs
1*n matrix dot with n*1
MP Model only uses integers as weights. After choosing weights we find threshold. Threshold is basically :-
After threshold (>=threshold) next node fires(1) and < threshold does not fire(0)
'''
w1=w2=1
mat_weight=np.matrix([[w1],[w2]])

mat_inputs_X=np.matrix([[0,0],[0,1],[1,0],[1,1]])
print(mat_inputs_X)
mat_weight_transpose=(mat_weight.getT())

dot_mat=(np.dot(mat_inputs_X,mat_weight))
dot_mat=list((dot_mat.getT()).flat)
#print(dot_mat)
##AND GATE
threshold=2
ans_mat=[]

for i in dot_mat:
	if(i<threshold):
		##Output 0 for AND Gate
		ans_mat.append(0)
	else:
		ans_mat.append(1)

print("------------------------")
print("AND GATE")
print("------------------------")
print("Input_1\tInput_2\tOutput")
print("-------\t-------\t------")


for i in range(len(ans_mat)):
	for j in (mat_inputs_X[i]):
		temp=list(j.flat)
		print("   "+str(temp[0])+"\t   "+str(temp[1])+"\t   "+str(ans_mat[i]))

##OR GATE

threshold=1
ans_mat=[]

for i in dot_mat:
	if(i<threshold):
		##Output 0 for AND Gate
		ans_mat.append(0)
	else:
		ans_mat.append(1)

print("------------------------")
print("OR GATE")
print("------------------------")
print("Input_1\tInput_2\tOutput")
print("-------\t-------\t------")
for i in range(len(ans_mat)):
	for j in (mat_inputs_X[i]):
		temp=list(j.flat)
		print("   "+str(temp[0])+"\t   "+str(temp[1])+"\t   "+str(ans_mat[i]))

###AND NOT GATE

##Updation of weights
w1=2;w2=-1
mat_weight=np.matrix([[w1],[w2]])
mat_weight_transpose=(mat_weight.getT())
dot_mat=(np.dot(mat_inputs_X,mat_weight))
dot_mat=list((dot_mat.getT()).flat)
threshold=2
ans_mat=[]

for i in dot_mat:
	if(i<threshold):
		##Output 0 for AND Gate
		ans_mat.append(0)
	else:
		ans_mat.append(1)

print("------------------------")
print("AND NOT GATE")
print("------------------------")
print("Input_1\tInput_2\tOutput")
print("-------\t-------\t------")
for i in range(len(ans_mat)):
	for j in (mat_inputs_X[i]):
		temp=list(j.flat)
		print("   "+str(temp[0])+"\t   "+str(temp[1])+"\t   "+str(ans_mat[i]))


### XOR Gate
### XOR gate will be formed using 2 layers :- x1*comp(x2)+comp(x1)*x2

## for first layer - 2 AND NOT GATES
w1=2;w2=-1
mat_weight_1=np.matrix([[w1],[w2]])
mat_weight_2=np.matrix([[w2],[w1]])
mat_weight_transpose_1=(mat_weight_1.getT())
mat_weight_transpose_2=(mat_weight_2.getT())

z1_dot_mat=(np.dot(mat_inputs_X,mat_weight_1))
z2_dot_mat=(np.dot(mat_inputs_X,mat_weight_2))
z1_dot_mat_1=list((z1_dot_mat.getT()).flat)
z2_dot_mat=list((z2_dot_mat.getT()).flat)
z1_ans_mat=[]
z2_ans_mat=[]


threshold=2
## ACTIVATION OF FIRST LAYER NODES

for i,j in zip(z1_dot_mat,z2_dot_mat):
	if(i<threshold):
		##Output 0 for AND Gate
		z1_ans_mat.append(0)
	else:
		z1_ans_mat.append(1)
	if(j<threshold):
		##Output 0 for AND Gate
		z2_ans_mat.append(0)
	else:
		z2_ans_mat.append(1)

print(z1_ans_mat);print(z2_ans_mat)
## for second layer - 1 OR GATE
threshold=1
w1=w2=1
mat_weight_final=np.matrix([[w1],[w2]])
z1_ans_mat=np.matrix(z1_ans_mat)
z2_ans_mat=np.matrix(z2_ans_mat)
combined_input=np.matrix(np.row_stack((z1_ans_mat,z2_ans_mat)))

final_dot_mat=np.dot(combined_input.getT(),mat_weight_final)
final_dot_mat=list((final_dot_mat.getT()).flat)
## FINAL ANSWER ACCORDING TO THRESHOLD
ans_mat=[]
print(final_dot_mat)

for i in final_dot_mat:
	if(i<threshold):
		##Output 0 for AND Gate
		ans_mat.append(0)
	else:
		ans_mat.append(1)

print("------------------------")
print("XOR GATE")
print("------------------------")
print("Input_1\tInput_2\tOutput")
print("-------\t-------\t------")
for i in range(len(ans_mat)):
	for j in (mat_inputs_X[i]):
		temp=list(j.flat)
		print("   "+str(temp[0])+"\t   "+str(temp[1])+"\t   "+str(ans_mat[i]))


