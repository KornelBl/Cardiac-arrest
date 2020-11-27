import numpy as np


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


def train_evaluate(X, Y, momentum_value, layer_size, train, test):

    mlp = MLPClassifier(hidden_layer_sizes=layer_size, solver="sgd", momentum=momentum_value, max_iter=1000, random_state=3, batch_size=100)

    mlp.fit(X[train], Y[train])

    predict = mlp.predict(X[test])

    score = mlp.score(X[test], Y[test])
    _confusion_matrix = confusion_matrix(Y[test], predict)

    return score, _confusion_matrix
