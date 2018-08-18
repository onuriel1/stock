import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def plot_dim_reduction(dataset, labels):
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(dataset)

    lda = LDA(n_components=2)
    lda_results = lda.fit_transform(dataset, labels)

    plot_data(pca_results, labels, "PCA of dataset")
    plt.show()

    plot_data(lda_results, labels, "LDA of dataset")
    plt.show()


def plot_data(X, labels, title=None):
    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange', 'purple', 'green', 'black', 'red', 'pink']
    classes = np.unique(labels)
    for color, i in zip(colors, classes):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], color = color, alpha = .8,
                    label = i)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title(title)


        # clf = Pipeline([
        #   ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
        #   ('classification', RandomForestClassifier())
        # ])
        # clf.fit(X, y)