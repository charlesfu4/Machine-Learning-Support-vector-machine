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
def a_extract(a):
	temp=[]
	for i in a:
		if i>10e-5:
			temp.append(i)
	return temp

def p_extract(a):
	index=numpy.where(a>10e-5)
	return index[0]

"minimizaiton process"
c=input("select C manually");

ret = minimize(objective, start, bounds=[(0,c) for b in range(n)], constraints={'type':'eq','fun':zerofunc})
alpha=ret['x']

print(alpha)

"Extract non-zero alpha"
alpha_ex=[]
sv=[]
targets_ex=[]

alpha_ex=a_extract(alpha)


print(p_extract(alpha))
for i in p_extract(alpha):
	targets_ex.append(targets[i])
	sv.append(inputs[i])

print(sv)

def b(s):
    temp=0
    for i in range(0,len(alpha_ex)):
        temp+=alpha_ex[i]*targets_ex[i]*kernel_lin([(sv[s])[0],(sv[s])[1]],[(sv[i])[0],(sv[i])[1]])
    temp=temp-targets_ex[s]
    return temp

def indicator(x,y):
    temp=0
    ind=0
    for i in range(0,len(alpha_ex)):
        temp+=alpha_ex[i]*targets_ex[i]*kernel_lin([x,y],[(sv[i])[0],(sv[i])[1]])
        ind=temp-b(0)
    return ind


"plotting process"
plt.plot([p[0] for p in classA],[p[1] for p in classA],'b. ')
plt.plot([p[0] for p in classB],[p[1] for p in classB],'r. ')

xgrid=numpy.linspace(-5,5)
ygrid=numpy.linspace(-4,4)
 
grid=numpy.array([[indicator(x,y) for x in xgrid] for y in ygrid])
 
plt.contour(xgrid,ygrid,grid,(-1.0,0.0,1.0),colors=('red','black','blue'),linewidths=(1,3,1))

plt.axis('equal')
plt.savefig('svmplot.pdf')
plt.show() 
