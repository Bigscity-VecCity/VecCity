
from logging import getLogger
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import csv
from veccity.downstream.downstream_models.abstract_model import AbstractModel


class RegressionModel(AbstractModel):

    def __init__(self, config):
        self._logger = getLogger()
        self.alpha = 1
        self.n_split = 5
        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.output_dim = config.get('output_dim', 32)
        self.exp_id = config.get('exp_id', None)
        self.result = {}
        self.result_path = './veccity/cache/{}/evaluate_cache/{}_evaluate_{}_{}_{}.csv'. \
            format(self.exp_id, self.exp_id, self.model, self.dataset, self.output_dim)

    def run(self,x,label):
        self._logger.info("--- Regression ---")
        x = np.array(x)
        label = np.array(label)
        kf = KFold(n_splits=self.n_split)
        y_preds = []
        y_truths = []
        for train_index, test_index in kf.split(x):
            train_index = list(train_index)
            test_index = list(test_index)
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = label[train_index], label[test_index]
            reg = linear_model.Ridge(alpha=self.alpha)
            X_train = np.array(X_train, dtype=float)
            y_train = np.array(y_train, dtype=float)
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            y_preds.append(y_pred)
            y_truths.append(y_test)

        y_pred[y_pred<0] = 0
        y_truths = np.concatenate(y_truths)
        y_preds = np.concatenate(y_preds)
        mae = mean_absolute_error(y_truths, y_preds)
        mse = mean_squared_error(y_truths, y_preds)
        rmse = mse**0.5
        r2 = r2_score(y_truths, y_preds)

        self.result={'mae':mae, 'mse':mse, 'r2':r2, 'rmse': rmse}
        self._logger.info(self.result)
        self.save_result(self.result_path)
        return self.result
    
    def clear(self):
        pass
    
    def save_result(self, save_path, filename=None):
        def dict_to_csv(dictionary, filename):
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=dictionary.keys())
                writer.writeheader()
                writer.writerow(dictionary)
        dict_to_csv(self.result, save_path)
        self._logger.info('Results is saved at {}'.format(save_path))