from sklearn.model_selection import train_test_split, KFold
import numpy as np

class CrossValidation(object):
    def __init__(self, data, n_fold, shuffle_seed, split_seed):
        self.n_fold = n_fold
        self.shuffle_seed = shuffle_seed
        self.split_seed = split_seed
        self.data=data

    def __iter__(self):
        np.random.seed(self.shuffle_seed)
        np.random.shuffle(self.data)
        data = np.array(self.data)
        kf = KFold(n_splits=self.n_fold,
                   random_state=self.split_seed, shuffle=True)
        for i, train_and_test_indexs in enumerate(kf.split(data)):
            train_indexs, test_indexs = train_and_test_indexs
            train_set = data[train_indexs]
            train_set, valid_set = train_test_split(
                train_set,
                test_size=1.0/9,
                shuffle=False
            )
            test_set = data[test_indexs]
            yield train_set,valid_set,test_set
            