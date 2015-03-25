from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import theano.tensor as T

from nn import NeuralNetworkClassifier, HiddenLayer


class ModelFactory(object):
    def create(self, **kwargs):
        steps = []
        if kwargs['scale']:
            steps.append(StandardScaler())
        if kwargs['pca']:
            steps.append(PCA(
                n_components=kwargs['pca_n_components'],
            ))

        if kwargs['model'] == 'knn':
            steps.append(KNeighborsClassifier(
                n_neighbors=kwargs['knn_n_neighbors'],
                p=kwargs['knn_p'],
            ))
        elif kwargs['model'] == 'nn':
            d_input = kwargs['pca_n_components'] if kwargs['pca'] else 28 ** 2
            n_hidden_units = kwargs['nn_n_hidden_units']
            d_output = 10
            steps.append(NeuralNetworkClassifier(
                layers=[
                    HiddenLayer(d_input, n_hidden_units, T.nnet.sigmoid),
                    HiddenLayer(n_hidden_units, d_output, T.nnet.softmax),
                ]
            ))
        else:
            raise ValueError('Unknown model')

        return make_pipeline(*steps)
