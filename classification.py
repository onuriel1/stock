import numpy as np
import itertools
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, AgglomerativeClustering, Birch,  FeatureAgglomeration, MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from data_object import DataObject
from sklearn.base import TransformerMixin, BaseEstimator
from tempfile import mkdtemp
from shutil import rmtree
import utils
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split as tts
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import  make_pipeline

DIM_REDUCTION_ALGORITHMS = [PCA, LinearDiscriminantAnalysis]
CLUSTERING_ALGORITHMS = [KMeans, DBSCAN, MeanShift, SpectralClustering, AffinityPropagation, GaussianMixture, AgglomerativeClustering, FeatureAgglomeration, Birch]

def identity(x):
    return x

def column_selector(columns, df):
    indexes = [index for index, col in enumerate(df.columns) if col in columns]
    def selector(x):
        if columns is None:
            return x
        return x[:,indexes]
    return selector

def balancer(method, *args, **kwargs):
    def balance(X, y):
        return method(*args, **kwargs).fit_sample(X, y)
    return balance

def create_function_transformer():
    return FunctionTransformer(pass_y=True)

#IGNORE_COLS = ['label', 'Mood', 'CPriceA', 'EPS', 'delta','id', 'zeroVsrest']
#IGNORE_COLS = ['label', 'Mood', 'CPriceA', 'EPS', 'delta','id', 'zeroVsrest', 'sma10', 'sma50', 'ema', 'rsi', 'macd', 'obv', 'accdist', 'atr', 'adx', 'stok', 'sto']
IGNORE_COLS = ['label', 'Mood', 'CPriceA', 'EPS', 'delta','id', 'zeroVsrest', 'symId', 'market_cap', 'sic', 'quarter', 'EPS_forecast']


# RandomUnderSampler.transform = RandomUnderSampler.sample
# RandomUnderSampler.fit_transform = RandomUnderSampler.fit_sample

def main():
    cachedir = mkdtemp()
    data_obj = DataObject("Vectors9", mood_threshold=3)
    data_obj.load_data()
    data = data_obj.get_data(not_cols=IGNORE_COLS,float_col_only=True)
    labels = data_obj.get_zero_vs_rest_labels()
    #x_train, x_test, y_train, y_test = tts(data, labels, random_state=42, test_size=0.1)
    x_train, x_test, y_train, y_test = tts(data, labels, test_size=0.1)
    rus = RandomUnderSampler()
    x_train, y_train = rus.fit_sample(x_train, y_train)
    pipes = []
    class_weights = {'class_weight': [{-1: 6, 1: 6}, {-1: 1, 1:1}, {-1: 10, 1: 10}]}
    # pipes.append(Pipe("balancer", "undersample", BalancingPipe, {'method': [RandomUnderSampler]}))
    pipes.append(Pipe("preprocess", 'pca', PCA, {"n_components" : [2,5,10]}))
    pipes.append(Pipe("preprocess", "identity", FunctionTransformer))
    #pyypipes.append(Pipe("classifier", "svm", LinearSVR))
    #pyypipes.append(Pipe("classifier", "svm", SVR))
    pipes.append(Pipe("classifier", "svm", LinearSVC))
    pipes.append(Pipe("classifier", 'lda', LinearDiscriminantAnalysis))
    # pipes.append(Pipe("feature_selection", 'column_selector', FunctionTransformer, {"func": [None, column_selector(['sic','EPS_forecast'], data)]}))
    pipes.append(Pipe('classifier', 'rf', RandomForestClassifier, {'criterion': ['gini','entropy'],'verbose': [10], "n_estimators" : [50, 75, 101, 151]}))
    pipes.append(Pipe("classifier", "knn", KNeighborsClassifier, {"n_neighbors" : [4, 8, 16, 32, 50,100,200,500, 700]}))
    steps = ['preprocess', "classifier"]
    gridcv = Pipe.create_pipeline(pipes, steps, memory=cachedir)
    gridcv.fit(x_train, y_train)

    data2, label2 = data_obj.get_non_zeros(IGNORE_COLS)
    x2_train, x2_test, y2_train, y2_test = tts(data2, label2, test_size=0.1)
    gridcv2 = Pipe.create_pipeline(pipes, steps, memory=cachedir)
    gridcv2.fit(x2_train, y2_train)
    anaylyze_predictor(gridcv, x_test, y_test)
    anaylyze_predictor(gridcv2, x2_test, y2_test)



def anaylyze_predictor(gridcv, x_test, y_test):
    print(gridcv.cv_results_)
    print("best score is ", gridcv.best_score_)
    print("using parameters :", gridcv.best_params_)
    y_pred = gridcv.predict(x_test)
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    print("accuracy on test: {0}".format(accuracy))



class Pipe:

    def __init__(self, step, name, method, parameters=None):
        self.step = step #step_name preprocess/classify
        self.name = name #SVM or randomforest
        self.method = method # RandomForestClassifier
        self.parameters = parameters # { n_estimators : [10,20]

    @staticmethod
    def create_pipeline(pipes, steps, cv=None, memory=None):
        pipeline_steps = {}
        all_params_grid = {}
        for pipe in pipes:
            if pipe.step in steps:
                index = steps.index(pipe.step)
                steps[index] = (pipe.step,pipe.method())
            if pipe.step in pipeline_steps:
                pipeline_steps[pipe.step].append(pipe.name)
            else:
                pipeline_steps[pipe.step] = [pipe.name]
            all_params_grid[pipe.name] = pipe.parameters if pipe.parameters is not None else {}
            all_params_grid[pipe.name]['object'] = pipe.method()
        param_grid = Pipe.make_param_grids(pipeline_steps, all_params_grid)
        pipeline = Pipeline(steps=steps, memory=memory)
        return GridSearchCV(pipeline, param_grid=param_grid, verbose=10, cv=cv, n_jobs=1)

    @staticmethod
    def make_param_grids(steps, param_grids):
        final_params = []

        # Itertools.product will do a permutation such that
        # (pca OR svd) AND (svm OR rf) will become ->
        # (pca, svm) , (pca, rf) , (svd, svm) , (svd, rf)
        for estimator_names in itertools.product(*steps.values()):
            current_grid = {}

            # Step_name and estimator_name should correspond
            # i.e preprocessor must be from pca and select.
            for step_name, estimator_name in zip(steps.keys(), estimator_names):
                for param, value in param_grids.get(estimator_name).items():
                    if param == 'object':
                        # Set actual estimator in pipeline
                        current_grid[step_name] = [value]
                    else:
                        # Set parameters corresponding to above estimator
                        current_grid[step_name + '__' + param] = value
            # Append this dictionary to final params
            final_params.append(current_grid)

        return final_params


class BalancingPipe(TransformerMixin, BaseEstimator):

    def __init__(self, method=RandomUnderSampler):
        TransformerMixin.__init__(self)
        BaseEstimator.__init__(self)
        self.method = method
        self.balancer = None

    def fit(self, X, y):
        self.balancer = self.method()
        self.balancer.fit(X, y)
        return self

    def transform(self, X, y):
        return self.balancer.sample(X, y)



if __name__=='__main__':
    main()

