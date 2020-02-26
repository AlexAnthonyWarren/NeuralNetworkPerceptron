# author = Alex Warren
# copyright = learnt from Polycode on youtube
# last edited = 16/2/20

import numpy as np

#neural network class
#sigmoid, sigmoidPrime, train, think
class NeuralNetwork():

    #constructor
    def __init__(self):

        np.random.seed(1)

        self.synampticWeights = 2 * np.random.random((3, 1)) - 1

    #sigmoid function
    #parameters x
    #return o(x)
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #sigmiod prime function - backward propagation
    # parameters x
    # return o'(x)
    def sigmoidPrime(self, x):
        return x * (1-x)

    #train
    #parameters traingingInputs, trainingOutputs, trainingIterations
    #iterate over trainingInputs as input for self.think
    #track dot product of trainingInputs and the output of self.think through sigmoidPrime - backwards prop
    #add these values to the synamptic weights, thus training the synaptic weights
    def train(self, trainingInputs, trainingOutputs, trainingIterations):

        for i in range(trainingIterations):
            outputs = self.think(trainingInputs)
            error = trainingOutputs - outputs

            #back propogation
            #dotprod
            adjustments = np.dot(trainingInputs.T, error * self.sigmoidPrime(outputs))

            self.synampticWeights += adjustments

    #think
    #parameters inputs
    #sigmoid the dot product of inputs and the trained synamptic weights, thus given and resulting output
    def think(self, inputs):

        inputs = inputs.astype(float)
        outputs = self.sigmoid(np.dot(inputs, self.synampticWeights))

        return outputs


if __name__ == "__main__":
    
    neuralNetwork = NeuralNetwork()
    print("Random starting synamptic weights:  ")
    print(neuralNetwork.synampticWeights)

    trainingInputs = np.array([[0,0,1],
                           [0,1,1],
                           [1,0,1],
                           [1,1,1]])

    trainingOutputs = np.array([[0,0,1,1]]).T

    neuralNetwork.train(trainingInputs, trainingOutputs, 50000)

    print("Synamptic weights after training")
    print(neuralNetwork.synampticWeights)

    counter = 0
    while True:
        print("\n")
        counter += 1
        print("Test" + str(counter))
        x1 = str(input("x1: "))
        x2 = str(input("x2: "))
        x3 = str(input("x3: "))

        print("New input set: " , x1, x2, x3)

        print("ouputs after training: ")
        print(neuralNetwork.think(np.array([x1, x2, x3])))

        
