import numpy, random, math
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

print(inputs)
print(targets)

"plotting process"


plt.plot([p[0] for p in classA],[p[1] for p in classA],'b. ')
plt.plot([p[0] for p in classB],[p[1] for p in classB],'r. ')
  
plt.axis('equal')
plt.savefig('svmplot.pdf')
plt.show() 
