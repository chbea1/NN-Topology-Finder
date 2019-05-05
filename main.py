from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random


MAX_LAYERS = 5
MAX_NEURONS = 64
POPULATION_EPOCHS =  10;
TRAINING_EPOCHS = 150;
POPULATION_SIZE = 20;

np.random.seed(7)

dataset = np.loadtxt("dataset.csv",delimiter=",")

X = dataset[:,0:8]
X_train = X[:len(X)//2]
X_validate = X[len(X)//2:]

Y = dataset[:,8]
Y_train = Y[:len(Y)//2]
Y_validate = Y[len(Y)//2:]

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class NeuralNetwork():
    def __init__(self,layers):
        self.layers = layers
        self.model = Sequential()
        self.model.add(Dense(layers[0], input_dim=8, activation='relu'))
        for layer in layers[1:-1]:
            self.model.add(Dense(layer,activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self,X,Y,epochs,batch=10):
        self.model.fit(X, Y, epochs=epochs, batch_size=batch,verbose=0)

    def evaluate(self,X,Y):
        scores = self.model.evaluate(X, Y)
        return scores[1] # accuracy

    def __str__(self):
        return str(self.layers)

    def __repr__(self):
        return self.__str__()


def evolution_round(population):
    survival_result = []
    for network in population:
        network.fit(X_train,Y_train,TRAINING_EPOCHS)
        score = network.evaluate(X_validate,Y_validate)
        survival_result.append( (network,score) )

    survival_result.sort(key = lambda tup: tup[1],reverse=True)
    return survival_result[:len(survival_result)//2]


def repopulate(population):
    newPop = []
    pairs = list(chunks(population,2))
    for pair in pairs:
        layers = (len(pair[0].layers) + len(pair[1].layers))//2
        if len(pair[0].layers) < len(pair[1].layers):
            left = pair[0].layers # Male female? No idea the proper name.
            right = pair[1].layers
        else:
            left = pair[1].layers
            right = pair[0].layers
        baby = []
        for i in range(0,len(left)):
            baby.append( (left[i] + right[i])//2 )
        baby = baby + right[layers:len(left)]

        newPop.append(pair[0])
        newPop.append(pair[1])
        newPop.append(NeuralNetwork(baby))
        newPop.append(NeuralNetwork(baby))
    return newPop


def evolve(population):
    for i in range(0,POPULATION_EPOCHS):

        survivals = evolution_round(population)
        print(survivals)
        survival_pop = list(list(zip(*survivals))[0])
        population = repopulate(survival_pop)
        print(population)





def run():
    population = []
    for i in range(0,POPULATION_SIZE):
        nn = NeuralNetwork(random.sample(range(1,MAX_NEURONS + 1),random.randint(1,MAX_LAYERS+1)))
        population.append(nn)


    evolve(population)






if __name__ == '__main__':
    run()

