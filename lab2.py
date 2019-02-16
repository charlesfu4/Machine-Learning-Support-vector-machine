import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt




def start(num_sam):
	return numpy.zeros(num_sam)

def kernel(data1,data2):
	return numpy.dot(input1,input2)


k=kernel()
ti=numpy.zeros(num_sam)
tj=numpy.zeros(num_sam)

def objective(alpha):
	pij=ti*tj*k
	return 

def zerofun():



ret = minimize()

alpha=['x']




"generating sample data points"
numpy.random.seed(100) "lock generated data"
classA = numpy.concatenate((numpy.random.randn(10,2)*0.2+[1.5,0.5],numpy.ramdon.randn(10,2)*0.2+[-1.5,0.5]))
classB = numpy.random.randn(20,2)*0.2+[0.0,-0.5]

inputs = numpy.concatenate((classA, classB))
targets = numpy.concatenate((numpy.ones(classA.shape[0]),-numpy.ones(classB.shape[0])))

N=inputs.shape[0] "number of rows"

permute=list(range(N))
random.shuffle(permute)
inputs=inputs[permute, :]
targets=targets[permute]


"plotting process"

plt.plot([p[0] for p in classA],[p[1] for p in classA],'b. ')
plt.plot([p[0] for p in classB],[p[1] ofr p in classB],'r. ')

plt.axis('equal')
plt.savefig('svmplot.pdf')
plt.show()



