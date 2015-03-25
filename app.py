from argparse import ArgumentParser

from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd

from dataset import DatasetFactory
from model import ModelFactory


class App(object):
    def cross_validation(self, model, training_set, n_folds):
        kf = KFold(training_set.size(), n_folds)
        training_scores = np.empty(n_folds)
        test_scores = np.empty(n_folds)
        for i, (training_indices, test_indices) in enumerate(kf):
            print('#{0}: '.format(i + 1), end='', flush=True)
            model.fit(
                training_set.training_data(training_indices),
                training_set.training_target(training_indices)
            )
            training_scores[i] = model.score(
                training_set.test_data(training_indices),
                training_set.test_target(training_indices)
            )
            test_scores[i] = model.score(
                training_set.test_data(test_indices),
                training_set.test_target(test_indices)
            )
            print('{0:f} (Training) / {1:f} (Test)'.format(
                training_scores[i],
                test_scores[i]
            ))
        print('Mean: {0:f} (Training) / {1:f} (Test)'.format(
            training_scores.mean(),
            test_scores.mean()
        ))

    def write_prediction(self, model, training_set, test_set, out):
        print('Fitting ... ', end='', flush=True)
        model.fit(training_set.training_data(), training_set.training_target())
        print('done.')
        print('Predicting ... ', end='', flush=True)
        prediction = model.predict(test_set.test_data())
        print('done.')
        df = pd.DataFrame(
            data=prediction,
            index=np.arange(1, test_set.size() + 1),
            columns=['Label'],
            dtype=int
        )
        df.to_csv(out, index_label='ImageId')


if __name__ == '__main__':
    parser = ArgumentParser(description='kaggle-digit-recognizer')
    parser.add_argument('--model')
    parser.add_argument('--training-size', default=42000, type=int)
    parser.add_argument('--test-size', default=28000, type=int)
    parser.add_argument('--cv', dest='cv', action='store_true')
    parser.add_argument('--no-cv', dest='cv', action='store_false')
    parser.set_defaults(cv=True)
    parser.add_argument('--cv-n-folds', default=5, type=int)
    parser.add_argument('--knn-n-neighbors', default=5, type=int)
    parser.add_argument('--knn-p', default=2, type=int)
    parser.add_argument('--nn-n-hidden-units', default=100, type=int)
    parser.add_argument('--augment', dest='augment', action='store_true')
    parser.add_argument('--no-augment', dest='augment', action='store_false')
    parser.set_defaults(augment=False)
    parser.add_argument('--augment-offset', default=0, type=int)
    parser.add_argument('--augment-length', default=28, type=int)
    parser.add_argument('--augment-slide', default=0, type=int)
    parser.add_argument('--scale', dest='scale', action='store_true')
    parser.add_argument('--no-scale', dest='scale', action='store_false')
    parser.set_defaults(scale=False)
    parser.add_argument('--pca', dest='pca', action='store_true')
    parser.add_argument('--no-pca', dest='pca', action='store_false')
    parser.set_defaults(pca=False)
    parser.add_argument('--pca-n-components', default=50, type=int)
    parser.add_argument('--output', default='knn.csv')
    kwargs = vars(parser.parse_args())
    print(kwargs)

    app = App()
    model = ModelFactory().create(**kwargs)
    training_set = DatasetFactory().create_training_dataset(**kwargs)
    if kwargs['cv']:
        app.cross_validation(model, training_set, kwargs['cv_n_folds'])
    else:
        test_set = DatasetFactory().create_test_dataset(**kwargs)
        app.write_prediction(model, training_set, test_set, kwargs['output'])

