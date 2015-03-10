from argparse import ArgumentParser

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from augment import *
from console import *
from dataset import *

if __name__ == '__main__':
    parser = ArgumentParser(description='k-Nearest Neighbors')
    parser.add_argument('--training-size', default=42000, type=int)
    parser.add_argument('--test-size', default=28000, type=int)
    parser.add_argument('--cv', dest='cv', action='store_true')
    parser.add_argument('--no-cv', dest='cv', action='store_false')
    parser.set_defaults(cv=False)
    parser.add_argument('--cv-n-folds', default=5, type=int)
    parser.add_argument('--knn-n-neighbors', default=5, type=int)
    parser.add_argument('--knn-p', default=2, type=int)
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
    parser.add_argument('--pca-n-components', default=28**2, type=int)
    parser.add_argument('--output', default='knn.csv')
    args = parser.parse_args()

    print(args)
    console = Console()
    training_dataset = TrainingDataset(size=args.training_size)
    if args.augment:
        augmenter = Augmenter(
            (28, 28),
            (args.augment_offset, args.augment_offset),
            (args.augment_length, args.augment_length),
            (args.augment_slide, args.augment_slide)
            )
        training_dataset = AugmentedTrainingDataset(training_dataset, augmenter)

    steps = []
    if args.scale:
        steps.append(StandardScaler())
    if args.pca:
        steps.append(PCA(
            n_components=args.pca_n_components,
            ))
    steps.append(KNeighborsClassifier(
        n_neighbors=args.knn_n_neighbors,
        p=args.knn_p,
        ))
    pipeline = make_pipeline(*steps)

    if args.cv:
        console.cross_validation(pipeline, training_dataset, args.cv_n_folds)
    else:
        test_dataset = TestDataset(size=args.test_size)
        if args.augment:
            test_dataset = AugmentedTestDataset(test_dataset, augmenter)
        console.write_prediction(pipeline, training_dataset, test_dataset, args.output)
