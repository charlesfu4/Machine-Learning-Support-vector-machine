import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

"generating sample data points" 
numpy.random.seed(100)
classA = numpy.concatenate((numpy.random.randn(10,2)*0.2+[1.5,0.5],numpy.random.randn(10,2)*0.2+[-1.5,0.5]))
classB = numpy.random.randn(20,2)*0.2+[0.0,-0.5]                                                                           
inputs = numpy.concatenate((classA, classB))
targets = numpy.concatenate((numpy.ones(classA.shape[0]),-numpy.ones(classB.shape[0])))
 
N=inputs.shape[0]
   
permute=list(range(N))
random.shuffle(permute)
inputs=inputs[permute, :]                                                                                                                
targets=targets[permute]

"the length of dataset array"
n=len(inputs)

"initial vector of guessing alpha"
start=numpy.zeros(n)

"kernal function define"
def kernel_lin(x1,x2):
	return numpy.dot(x1,x2)

def kernel_poly(x1,x2,p=2):
	return numpy.power((numpy.dot(x1,x2)+1),p)

def kernel_rad(x1,x2,sig=2):
	return math.exp(-numpy.dot((numpy.substract(x1,x2)),(numpy.substract(x1,x2)))/(2*sig^2))

"Compute Pij as a golbal variable"
def Pij(data,t,kernel):
	P=numpy.zeros((n,n))
	for i in range(0,n):
		for j in range(0,n):
			P[i][j]=t[i]*t[j]*kernel([(data[i])[0], (data[i])[1]], [(data[j])[0], (data[j])[1]])
	return P

P_ij=Pij(inputs,targets,kernel_lin)


"Implement objective funciton"
def objective(a):
	return 0.5 * numpy.dot(a,numpy.dot(a,P_ij))-numpy.sum(a)

"zerofunction"
def zerofunc(a):
	return numpy.dot(a,targets)
				  
"extract funciton"
def extract(a):
	temp=[]
	for i in a:
		if i>10e-5:
			temp.append(i)
	return temp


"minimizaiton process"
c=input("select C manually");

ret = minimize(objective, start, bounds=[(0,c) for b in range(n)], constraints={'type':'eq','fun':zerofunc})
alpha=ret['x']

print(alpha)

"Extract non-zero alpha"
extarcts=[]
extarcts=extract(alpha)
print(extarcts)



"plotting process"
plt.plot([p[0] for p in classA],[p[1] for p in classA],'b. ')
plt.plot([p[0] for p in classB],[p[1] for p in classB],'r. ')
  
plt.axis('equal')
plt.savefig('svmplot.pdf')
plt.show() 
