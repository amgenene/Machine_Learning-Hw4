from cvxopt import solvers, matrix
import math
import numpy as np
import sklearn.svm
np.set_printoptions(precision=20)

class SVM453X ():
    def __init__ (self):
        pass

    # Expects each *row* to be an m-dimensional row vector. X should
    # contain n rows, where n is the number of examples.
    # y should correspondingly be an n-vector of labels (-1 or +1).
    def fit (self, X, y):
    #     # TODO change these -- they should be matrices or vectors
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((X, ones))
        G = (-y * X.T).T
        P = np.eye(X.shape[1])
        q = np.zeros(X.shape[1])
        h = np.ones(y.shape[0])
        h *= -1
    #     #[-yi*Xi, -yi]*[w /n, B]
    #     # Solve -- if the variables above are defined correctly, you can call this as-is:
        sol = solvers.qp(matrix(P, tc='d'), matrix(q, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'))
    #     # Fetch the learned hyperplane and bias parameters out of sol['x']
        self.w = np.array(sol['x'])[0:-1, :].T  # TODO change this
        self.b = np.array(sol['x'])[-1, :]  # TODO change this
    # # Given a 2-D matrix of examples X, output a vector of predicted class labels
    def predict (self, x):
        predictions = x.dot(self.w.T) + self.b
        predictions[predictions >= 0] = 1
        predictions[predictions < 0] = -1
        return predictions.T # TODO fix

def test1 ():
    # Set up toy problem
    X = np.array([ [1,1], [2,1], [1,2], [2,3], [1,4], [2,4] ])
    y = np.array([-1,-1,-1,1,1,1])
    # Train your model
    svm453X = SVM453X()
    svm453X.fit(X, y)
    print(svm453X.w, svm453X.b)
    # Compare with sklearn
    svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard-margin
    svm.fit(X, y)
    print(svm.coef_, svm.intercept_)
    print(svm453X.predict(X), svm.predict(X))
    acc = np.mean(svm453X.predict(X) == svm.predict(X))
    print("Acc={}".format(acc))

def test2 (seed):
    np.random.seed(seed)
    # Generate random data
    X = np.random.rand(20,3)
    # Generate random labels based on a random "ground-truth" hyperplane
    while True:
        w = np.random.rand(3)
        y = 2*(X.dot(w) > 0.5) - 1
        # Keep generating ground-truth hyperplanes until we find one
        # that results in 2 classes
        if len(np.unique(y)) > 1:
            break

    svm453X = SVM453X()
    svm453X.fit(X, y)

    # Compare with sklearn
    svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard margin
    svm.fit(X, y)
    diff = np.linalg.norm(svm.coef_ - svm453X.w) + np.abs(svm.intercept_ - svm453X.b)
    print(diff)

    acc = np.mean(svm453X.predict(X) == svm.predict(X))
    print("Acc={}".format(acc))

    if acc == 1 and diff < 1e-1:
        print("Passed")

if __name__ == "__main__":
    test1()
    for seed in range(5):
        test2(seed)
