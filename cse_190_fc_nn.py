import numpy as np

class multilayer_nn(object):
    def __init__(self, layer_sizes, activation_funcs, derivative_funcs):
        """
        :param activation_func: The activation function (ex: logistic, perceptron, softmax).
        :param n_inputs: Number of features to classify based on.
        :param n_outputs: Number of categories to classify data into.

        """
        self.w = []
        self.a_func = activation_funcs
        self.d_a_func = derivative_funcs
        self.layer_sizes = layer_sizes

        self.output_layer = len(layer_sizes)-2
        np.random.seed(seed = 1)

        for i in range(0,len(layer_sizes)-1):
            self.w.append(np.random.normal(0, 0.1, size=(layer_sizes[i], layer_sizes[i + 1])))

        self.w = np.array(self.w)
        self.velocity = np.zeros_like(self.w)
        
    def propogate(self,inputs,layer):
        return self.a_func[layer](np.dot(inputs, self.w[layer]))

    def train_batch(self, inputs, labels, l_rate = .00001, momentum = 0, l1 = 0, l2 = 0):
        deltas = np.zeros_like(self.w)
        updates = np.zeros_like(self.w)
        # forward propogate
        outputs = [] # outputs[n] is the output of the nth layer of the NN
        outputs.append(self.propogate(inputs,0)) # find the output of layer 1
        
        for i in range(1, len(self.w)):
            outputs.append(self.propogate(outputs[i-1],i)) # find the output of subsequent layers

        error = labels - outputs[self.output_layer] # find the errors
        
        deltas[self.output_layer] = error * l_rate # Error for last layer
        for i in range(self.output_layer-1,-1,-1): # Propogate the error backwards
            deltas[i] = np.dot(deltas[i+1],self.w[i+1].T) * (self.d_a_func[i](outputs[i]))

        #update the weights from the errors of each neuron and the their inputs
        updates[0] = np.dot(inputs.T,deltas[0]) + self.velocity[0] - l1 * np.sign(self.w[0]) - l2 * self.w[0]
        self. velocity[0] = momentum * updates[0]
        self.w[0] += updates[0]
        for i in range(1,self.output_layer+1):
            updates[i] = np.dot(outputs[i - 1].T,deltas[i])  + self.velocity[i] - l1 * np.sign(self.w[i]) - l2 * self.w[i]
            self.velocity[i] = momentum * updates[i]
            self.w[i] += updates[i]

        MSE = np.var(error)
        print "MSE = ", MSE



def sigmoid(x):
    # overflow safe
    if(x > 500):
        return 1
    if(x < -500):
        return 0
    else:
        return 1 / (1 + np.exp(-x))  # activation function
sigmoid = np.vectorize(sigmoid)
def d_sigmoid(x): return x * (1 - x)  # derivative of sigmoid

def identity(x): return x
def d_identity(x): return 1

def softmax(x):
    def safe_exp(x):
        if (x > -500):
            return np.exp(x)
        else:
            return 0  # derivative of sigmoid
    safe_exp = np.vectorize(safe_exp)
    #overflow safe
    out = np.zeros_like(x)
    for i in range(len(x)):
        e = safe_exp(x[i]- np.max(x[i]))
        out[i] = e / e.sum()
    return out



def relu(x): return max(x,0) # activation function
relu = np.vectorize(relu)

def d_relu(x):
    if (x > 0):
        return 1
    else:
        return 0  # derivative of sigmoid
d_relu = np.vectorize(d_relu)

def tanh(x):
    return 2 * sigmoid(2*x) - 1
def d_tanh(x):
    return 4 * d_sigmoid(2*x)

#X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#t = np.array([[0,1],[1,0],[1,0],[0,1]])

#NN = multilayer_nn([2,5,2], [sigmoid,softmax],[d_sigmoid])
#NN2 = multilayer_nn([sigmoid,identity],[d_sigmoid,identity],[2,3,1])


#for i in range(2000000):
 #   NN.train_batch(X, t, 1)
