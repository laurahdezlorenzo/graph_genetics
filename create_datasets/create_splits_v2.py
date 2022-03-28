import pandas as pd
from sklearn.model_selection import StratifiedKFold

def create_folds_stratified_cv(n):

    skf = StratifiedKFold(n_splits=n)
    for train, test in skf.split(X, y)


if __name__ == '__main__':

    target = 'DX'
