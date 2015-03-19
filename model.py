from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
        else:
            raise ValueError('Unknown model')

        return make_pipeline(*steps)
