from sklearn.base import ClassifierMixin
import numpy as np
import scipy
import theano
import theano.tensor as T


class HiddenLayer(object):
    def __init__(self, d_input, d_output, activation):
        self.d_input = d_input
        self.d_output = d_output
        self.activation = activation

        self.weights = theano.shared(
            np.random.uniform(-1, 1, (d_input, d_output)),
            'weights',
            borrow=True
        )
        self.biases = theano.shared(
            np.random.uniform(-1, 1, d_output),
            'biases',
            borrow=True
        )
        self.params = (self.weights, self.biases)

    def forward(self, inputs):
        return self.activation(T.dot(inputs, self.weights) + self.biases)

    def get_params(self):
        return np.concatenate((
            self.weights.get_value().flatten(),
            self.biases.get_value().flatten(),
        ))

    def set_params(self, params):
        shape = self.weights.get_value().shape
        for param in (self.weights, self.biases):
            shape = param.get_value().shape
            param.set_value(params[:np.prod(shape)].reshape(shape))
            params = params[np.prod(shape):]
        return params


class NeuralNetworkClassifier(ClassifierMixin):
    def __init__(self, layers):
        self.layers = layers

        self.inputs = self.outputs = T.matrix('inputs')
        for layer in layers:
            self.outputs = layer.forward(self.outputs)
        self.outputs_f = theano.function([self.inputs], self.outputs)

        self.params = ()
        for layer in layers:
            self.params += layer.params

    def fit(self, X, y):
        cost = T.sum(T.nnet.categorical_crossentropy(
            self.outputs,
            T.extra_ops.to_one_hot(y, np.unique(y).size)
        ))
        cost_grad = T.grad(cost, self.params)

        cost_f = theano.function([self.inputs], cost)
        cost_grad_f = theano.function([self.inputs], cost_grad)

        params = scipy.optimize.fmin_cg(
            lambda params: (self.set_params(params), cost_f(X))[1],
            self.get_params(),
            lambda params: (self.set_params(params), self.flatten_params(cost_grad_f(X)))[1]
        )

    def predict(self, X):
        return np.argmax(self.predict_proba(X), 1)

    def predict_proba(self, X):
        return self.outputs_f(X)

    def flatten_params(self, tensors):
        return np.concatenate([tensor.flatten() for tensor in tensors])

    def get_params(self):
        return np.concatenate([layer.get_params().flatten() for layer in self.layers])

    def set_params(self, params):
        for layer in self.layers:
            params = layer.set_params(params)


if __name__ == '__main__':
    convnet = NeuralNetworkClassifier([
        HiddenLayer(2, 10, T.nnet.sigmoid),
        HiddenLayer(10, 10, T.nnet.sigmoid),
        HiddenLayer(10, 2, T.nnet.softmax),
    ])
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    y = np.array([
        1,
        0,
        0,
        1,
    ])
    convnet.fit(X, y)
    print(convnet.predict(X))
