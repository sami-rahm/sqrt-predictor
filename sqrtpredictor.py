import random
import math
import matplotlib.pyplot as plt

class neural_network:
    def __init__(self,inp,h,o):#initialise network
        self.inputs=[1 for _ in range(inp)]
        self.hidden=[[[1,0] for _ in range(h[i])] for i in range(len(h))]#neuron value,bias
        self.outputs=[[1,0] for _ in range(o) ]
        self.weights=[]

        self.weights.append([[random.uniform(-1,1) for i in range(h[0])] for _ in range(inp)])

        for i in range(len(h)-1):
            self.weights.append([[random.uniform(-1,1) for n in range(h[i+1])] for w in range(h[i])])

        self.weights.append([[1 for i in range(o)] for _ in range(h[-1])]) 
        #layer indexing is [layer][source neuron][target neuron]
       
    def sigmoid(self,n):#sigmoid activation
        return 1/(1+2.718**(-n))
    def sigmoid_derivative(self,n):
        return self.sigmoid(n)*(1-self.sigmoid(n))
    def mse(self,predval,realval):#mean squared error
        return (realval-predval)**2
    def mse_derivative(self,predval,realval):
        return 2*(predval-realval)
   
    def forward(self,inputvalues):#forward propagation- calculates outputs
        if len(inputvalues)!=len(self.inputs):
            raise ValueError("input size doesnt match input network size")
        self.inputs=inputvalues
        for h in range(len(self.hidden[0])):#iterating through neurons in hidden layer 0
            nsum=0#sum of all input neurons * weight connected to it
            for n in range(len(self.inputs)):#iterating through input neurons
                nsum+=self.inputs[n]*self.weights[0][n][h]
            nsum=self.sigmoid(nsum+self.hidden[0][h][1])#adds the bias then applies sigmoid activation
            self.hidden[0][h][0]=nsum
        if len(self.hidden)!=1:
            for layer in range(len(self.hidden)-1):
                for h in range(len(self.hidden[layer+1])):
                    nsum=0
                    for n in range(len(self.hidden[layer])):
                        nsum+=self.hidden[layer][n][0]*self.weights[layer+1][n][h]
                    nsum=self.sigmoid(nsum+self.hidden[layer+1][h][1])
                    self.hidden[layer+1][h][0]=nsum
        output_values=[0 for i in range(len(self.outputs))]
        for o in range(len(self.outputs)):#calculate output layer
            nsum=0
            for n in range(len(self.hidden[-1])):
                nsum+=self.hidden[-1][n][0]*self.weights[-1][n][o]
            nsum=nsum+self.outputs[o][1]
            self.outputs[o][0]=nsum
            output_values[o]=nsum
        return output_values
    
    def backward(self,target_values):#backpropagation one pass
        learning_rate=0.1#how quickly it converges
        inputs=self.inputs
        pred_values=self.forward(inputs)
        values_length=len(pred_values)
        if len(target_values)!=values_length:
            raise ValueError("target value size doesnt match predicted values size")
        errors=[self.mse(pred_values[i],target_values[i]) for i in range(values_length)]
        def calculategradients():
            #this is only temporary for last layer only will change soon
            error_derivatives=[self.mse_derivative(pred_values[i],target_values[i]) for i in range(values_length)]
           
            #partial derivative with respect to weights in last layer
            llayer_dx=[[0 for _ in range(len(self.outputs))]for _ in range(len(self.hidden[-1]))]
            #partial derivative with respect to weights in second last layer
            if len(self.hidden)>=2:
                self.testlayer=[self.hidden[-2][n][0] for n in range(len(self.hidden[-2]))]
                slayer_dx=[[0 for _ in range(len(self.hidden[-1]))]for _ in range(len(self.hidden[-2]))]
            else:
                self.testlayer=self.inputs
                slayer_dx=[[0 for _ in range(len(self.hidden[-1]))]for _ in range(len(self.inputs))]
            sblayer_dx=[ sum(error_derivatives[j]*self.weights[-1][out][j] for j in range(len(self.outputs)))\
                        *self.hidden[-1][out][0] *(1-self.hidden[-1][out][0]) for out in range(len(self.hidden[-1]))]
            #calculate partial derivatives for weights in last layer
            for i in range(len(self.hidden[-1])):
                for j in range(len(self.outputs)):
                    #indexing [hiddenneuron][output]
                    llayer_dx[i][j]=error_derivatives[j]*self.hidden[-1][i][0]
            for inp in range(len(self.testlayer)):
                for out in range(len(self.hidden[-1])):
                    #indexing [hiddenneuron][output]
                    slayer_dx[inp][out]=sum(error_derivatives[j]*self.weights[-1][out][j] for j in range(len(self.outputs)))\
                        *self.testlayer[inp]*self.hidden[-1][out][0] *(1-self.hidden[-1][out][0])
 
            return llayer_dx,slayer_dx,error_derivatives,sblayer_dx

        gradl,grads,gradbl,gradbsl=calculategradients()
        #perform one backwards pass on every weight in the last layer ONLY TEMPORARY for debugging
        for i in range(len(self.hidden[-1])):
            for j in range(len(self.outputs)):
                self.weights[-1][i][j]-=gradl[i][j]*learning_rate
        for o in range(len(self.outputs)):self.outputs[o][1]-=gradbl[o]*learning_rate
        #perform one backwards pass on every weight in the penultimate layer ONLY TEMPORARY for debugging
        for inp in range(len(self.testlayer)):
            for out in range(len(self.hidden[-1])):
                self.weights[-2][inp][out]-=grads[inp][out]*learning_rate
        for l in range(len(self.hidden[-1])):self.hidden[-1][l][1]-=gradbsl[l]*learning_rate
    def trainnet(self,target_values,iterations):     
        for i in range(iterations):
            self.backward(target_values)
            if i//100==i/100:#if 100 iterations have passed print error
                totalerror=0
                for n in range(len(target_values)):
                    totalerror+=self.mse(self.forward(self.inputs)[n],target_values[n]) 
                print("loss for iterations:",i,":",totalerror)
   
#____interface- not related to neuralnetworkclass___________
nn=neural_network(1,[10],1)
inputvals=[i for i in range(1,101)]
targetvals=[math.sqrt(i) for i in range(1,101)]
predictedvals=[0 for _ in range(100)]
for i in range(1000):
    for x in range(len(inputvals)):
        nn.forward([inputvals[x]/100])
        nn.backward([targetvals[x]])
    if i%100==0:
        print("iteration:",i)
       
for n in range(len(inputvals)):
     predictedvals[n]=(nn.forward([inputvals[n]/100])[0])
     print(inputvals[n],predictedvals[n])

plt.figure(figsize=(8, 6))
plt.plot(inputvals, targetvals, marker='o', linestyle='-', color='b', label=r'$y = \sqrt{x}$')
plt.plot(inputvals, predictedvals, marker='s', linestyle='--', color='r', label=r'predicted values')
# Labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Graph of $y = \sqrt{x}$')

# Show grid and legend
plt.grid(True)
plt.legend()

# Display the graph
plt.show()
#_________________________________________________________
