'''
1.load data
2.random chose data or odd&even to make training data(50%)
remember to normalize data with /max
3.calculate M and Sigma of each class (M for mean vector;Sigma for covariance matrix)
4.MAP classification
5.professor's request
	(1)training accuracy
	(2)test accuracy
	(3)cost time
'''
import numpy as np
from numpy.linalg import inv
import time
import math
#from collections import Counter

def create_trainset(data):
#select 50% data to be training data
	trainlist = []
	for p in range(0,np.size(data,0)):#np.size(data,0)=#row
		if p %2==0:
			trainlist.append(p)
	#for sigma,mean, must to keep class feature
	#data=np.delete(data,0,1)#delete class
	trainset=data.take(trainlist,axis=0)
	
	return trainset
	
def create_novelset(data):
#select 50% data to be novel data
	novellist = []
	for p in range(0,np.size(data,0)):#np.size(data,0)=#row
		if p %2==1:
			novellist.append(p)
	#data=np.delete(data,0,1)#delete class
	novelset=data.take(novellist,axis=0)
	
	return novelset

def slice_data_by_class(data):
#slice data
	list_1=[]
	list_2=[]
	list_3=[]
	for p in range(0,np.size(data,0)):#np.size(data,0)=#row
		if data[p,0]==1:
			list_1.append(p)
		elif data[p,0]==2:
			list_2.append(p)		
		elif data[p,0]==3:
			list_3.append(p)
	data=np.delete(data,0,1)
	arr_1=data.take(list_1,axis=0)
	arr_2=data.take(list_2,axis=0)
	arr_3=data.take(list_3,axis=0)	
	return [arr_1,arr_2,arr_3]

def mean_vector(data):
#calculate mean vector;dim=13
	return np.mean(data,axis=0)

def covariance_matrix(data):
#calculate covariance matrix
	trans=np.transpose(data)
	return np.cov(trans)

def MAP_classify(input_sample,M,Sigma,P):
#operate MAP classification
#Assume the data is Gaussion distribution
#h(x)=1/2*(X-M1)^T*Sigma1^(-1)*(X-M1)-1/2*(X-M2)^T*Sigma2^(-1)*(X-M2)+1/2*ln(|Sigma1|/|Sigma2|)
#<>ln(P2/P1)
#matrix multiply=np.matmul(a, b)
#nature log=math.log(x)
#determinat of matrix=np.linalg.det(a)
	h=[]
	s={}
	p={}

	for i in range(len(P)):
		h.append(np.transpose((np.subtract(input_sample,M[i]))))
		h[i]=np.matmul(h[i],inv(np.matrix(Sigma[i])))
		h[i]=np.matmul(h[i],np.subtract(input_sample,M[i]))
		
	#s[1]=s12=math.log(((np.linalg.det(Sigma[0]))/(np.linalg.det(Sigma[1]))))
	#s[2]=s13=math.log(((np.linalg.det(Sigma[0]))/(np.linalg.det(Sigma[2]))))
	#s[3]=s23=math.log(((np.linalg.det(Sigma[1]))/(np.linalg.det(Sigma[2]))))

	#p[1]=p12=2*math.log(P[1]/P[0])
	#p[2]=p13=2*math.log(P[2]/P[0])
	#p[3]=p23=2*math.log(P[2]/P[1])
	for i in range(len(P)-1):
		if i==0:
			s[i+i+2]=math.log(((np.linalg.det(Sigma[i]))/(np.linalg.det(Sigma[i+2]))))
			p[i+i+2]=2*math.log(P[i+2]/P[i])
		s[i+i+1]=math.log(((np.linalg.det(Sigma[i]))/(np.linalg.det(Sigma[i+1]))))
		p[i+i+1]=2*math.log(P[i+1]/P[i])

	temp={}
	for i in range(len(P)-1):
		if i==0:
			temp[i+i+2]=h[i]-h[i+2]+s[i+i+2]
		temp[i+i+1]=h[i]-h[i+1]+s[i+i+1]

	#compare temp with p
	if temp[1]<p[1]:#class_1<class_2
		if temp[2]<p[2]:#class_1<class_3
			result=1
		else:#class_3<class_1
			result=3
	else:#class_2<class_1
		if temp[3]<p[3]:#class_2<class_3
			result=2
		else:#class_3<class_2
			result=3

	return result

def accuracy(testset,trainset,M,Sigma,P):
#calculate accuracy of classification
	start_time=time.time()
	rate=len(testset)

	for p in range(0,np.size(testset,0)):#np.size(data,0)=#row
		testsample=testset[p][1:]
		np.delete(data,0,1)#delete class
		classify_result=MAP_classify(testsample,M,Sigma,P)
		#right do nothing(100%),wrong minus
		if classify_result!=testset[p,0]:
			rate-=1

	cost=time.time()-start_time	
	return (rate/len(testset),cost)

#main function
if __name__ == '__main__':
	#load source data
	data={}
	data_source=open('wine.data.txt','r')
	i=0
	for line in data_source:
		features=line.strip('\n').split(',')#list
		data[i]=[int(features[0])]
		del features[0]
		for f in features:
			data[i].append(float(f))
		i+=1
	data_source.close()
	
	#initialization
	arr=np.array(data[0])
	#row
	for j in range(1,len(data)):
		arr=np.vstack((arr,np.array(data[j])))

	#start classify
	trainset = create_trainset(arr)
	novelset = create_novelset(arr)
	
	#calculate mean vector and covariance matrix
	arr_class=slice_data_by_class(trainset)
	
	M=[]
	Sigma=[]
	P=[]
	for i in range(0,3):
		M.append(mean_vector(arr_class[i]))
		Sigma.append(covariance_matrix(arr_class[i]))
		P.append(len(arr_class[i]))

	train_result=accuracy(trainset,trainset,M,Sigma,P)
	print('training set accuracy: '+str(train_result[0]*100)+'%')
	print('total cost: '+str(train_result[1]))
	novel_result=accuracy(novelset,trainset,M,Sigma,P)
	print('novel set accuracy: '+str(novel_result[0]*100)+'%')
	print('total cost: '+str(novel_result[1]))
	
	