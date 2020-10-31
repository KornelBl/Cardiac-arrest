import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


def train_evaluate(X, Y, momentum_value, layer_size):
    rkf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)

    mlp = MLPClassifier(hidden_layer_sizes=layer_size, max_iter=1000, momentum=momentum_value)

    confusion_matrix_sum = np.zeros(shape=(5, 5))
    score_sum = 0
    for train, test in rkf.split(X, Y):

        mlp.fit(X[train], Y[train])

        predict = mlp.predict(X[test])

        score = mlp.score(X[test], Y[test])
        temp_confusion_matrix = confusion_matrix(Y[test], predict)

        confusion_matrix_sum += temp_confusion_matrix
        score_sum += score

    score_sum /= 10
    confusion_matrix_sum /= 10

    return score_sum, confusion_matrix_sum


def save_results_to_csv(scores, confusion_matrix, filename: str = None):


