from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd


class Console(object):
    def cross_validation(self, classifier, training_dataset, n_folds):
        kf = KFold(training_dataset.size(), n_folds)
        scores = np.empty(n_folds)
        for i, (training_indices, test_indices) in enumerate(kf):
            print('#{0}: '.format(i + 1), end='', flush=True)
            classifier.fit(
                training_dataset.training_data(training_indices),
                training_dataset.training_target(training_indices)
                )
            prediction = classifier.predict(training_dataset.test_data(test_indices))
            target = training_dataset.test_target(test_indices)
            error = np.zeros([10, 10], dtype=int)
            for j in range(len(test_indices)):
                error[target[j], prediction[j]] += 1;
            scores[i] = 100 * (1 - error.trace() / len(test_indices))
            print('{0:f}'.format(scores[i]))
            print(error)
            print()
        print('Mean: {0:f}'.format(scores.mean()))
        return scores.mean()

    def write_prediction(self, classifier, training_dataset, test_dataset, out):
        print('Fitting ... ', end='', flush=True)
        classifier.fit(training_dataset.training_data(), training_dataset.training_target())
        print('done.')
        print('Predicting ... ', end='', flush=True)
        prediction = classifier.predict(test_dataset.test_data())
        print('done.')
        df = pd.DataFrame(
            data=prediction,
            index=np.arange(1, test_dataset.size() + 1),
            columns=['Label'],
            dtype=int
            )
        df.to_csv(out, index_label='ImageId')
