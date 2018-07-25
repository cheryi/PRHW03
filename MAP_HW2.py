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

def generate_gaussian(dim,N):
	sample_matrix=[]
	for node in range(N):
		current_vector = []
		for j in range(dim):
			r = np.random.normal(loc=0, scale=1, size=None)#get 1 gaussion
			current_vector.append(r)
		sample_matrix.append(current_vector)
	return sample_matrix
	
def generate_data(dim,p,N):#p refers to sigma
	known = generate_gaussian(dim,N)
	mean = np.zeros((dim,1))
	#mean = np.matrix([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
	#produce covariance matrix
	cov=[]
	for i in range(dim):
		row=[]
		for j in range(dim):
			row.append(math.pow(p,abs(i-j)))
		cov.append(row)
	#transfer to ndarray
	cov_m=np.asarray(cov)

	[eigenvalues, eigenvectors]=np.linalg.eig(cov_m)
	lamda=np.matrix(np.diag(np.sqrt(eigenvalues)))
	Q=np.matrix(eigenvectors)*lamda
	control=0
	for each in known:
		original=np.matrix(each).copy().transpose()
		tweaked=(Q*original)+mean
		tweaked = np.transpose(tweaked)
		#insert class
		tweaked = np.insert(tweaked,0,p)
		if control!=0:
			tweaked_all=np.concatenate((tweaked_all,tweaked))
		else:
			tweaked_all=tweaked
			control+=1
	return tweaked_all

def create_trainset(data):
#select 50% data to be training data
	trainlist = []
	for p in range(0,np.size(data,0)):#np.size(data,0)=#row
		if p %2==0:
			trainlist.append(p)
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
#slice data by class
	list_1=[]
	list_2=[]
	for p in range(0,np.size(data,0)):#np.size(data,0)=#row
		if data[p,0]==0.9:
			list_1.append(p)
		elif data[p,0]==0.5:
			list_2.append(p)
	data=np.delete(data,0,1)
	arr_1=data.take(list_1,axis=0)
	arr_2=data.take(list_2,axis=0)
	return [arr_1,arr_2]	

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
	input_sample=np.transpose(input_sample)
	
	for i in range(len(P)):
		h.append(np.transpose((np.subtract(input_sample,M[i]))))
		h[i]=np.matmul(h[i],inv(np.matrix(Sigma[i])))
		h[i]=np.matmul(h[i],np.subtract(input_sample,M[i]))
	
	#s=s12
	s=math.log(((np.linalg.det(Sigma[0]))/(np.linalg.det(Sigma[1]))))
	#p=p12
	p=2*math.log(P[1]/P[0])
	temp=h[0]-h[1]+s
	#compare temp with p
	if temp<p:
		result=0.9
	else:
		result=0.5
	return result

def accuracy(testset,trainset,M,Sigma,P):
#calculate accuracy of classification
	start_time=time.time()
	rate=len(testset)

	for p in range(0,np.size(testset,0)):#np.size(data,0)=#row
		testsample=testset[p,1:]
		classify_result=MAP_classify(testsample,M,Sigma,P)
		
		#right do nothing(100%),wrong minus
		if classify_result!=testset[p,0]:
			rate-=1

	cost=time.time()-start_time	
	return (rate/len(testset),cost)

#main function
if __name__ == '__main__':
	#generate data
	c1=generate_data(20,0.9,100)
	c2=generate_data(20,0.5,100)
	data=np.concatenate((c1,c2))
	
	#start classify
	trainset = create_trainset(data)
	novelset = create_novelset(data)
	
	#calculate mean vector and covariance matrix
	arr_class=slice_data_by_class(trainset)
	
	M=[]
	Sigma=[]
	P=[]
	for i in range(0,2):
		M.append(mean_vector(arr_class[i]))
		Sigma.append(covariance_matrix(arr_class[i]))
		P.append(len(arr_class[i]))
	#must transpose M	
	for i in range(len(M)):
		M[i]=np.transpose(M[i])

	train_result=accuracy(trainset,trainset,M,Sigma,P)
	print('training set accuracy: '+str(train_result[0]*100)+'%')
	print('total cost: '+str(train_result[1]))
	novel_result=accuracy(novelset,trainset,M,Sigma,P)
	print('novel set accuracy: '+str(novel_result[0]*100)+'%')
	print('total cost: '+str(novel_result[1]))
	
	