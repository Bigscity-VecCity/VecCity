from logging import getLogger
from sklearn.metrics import  mean_squared_error, mean_absolute_error
import numpy as np

from veccity.downstream.downstream_models.abstract_model import AbstractModel
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from veccity.data.dataset.dataset_subclass.bert_vocab import WordVocab
from veccity.data.preprocess import cache_dir
import os
import random


class SpeedInferenceModel(AbstractModel):
    def __init__(self, config):
        self._logger = getLogger()
        self.alpha = config.get('alpha', 1)
        self.n_split = config.get('n_split', 5)
        self.exp_id = config.get('exp_id', None)
        self.result_path = './veccity/cache/{}/evaluate_cache/regression_{}_{}.npy'. \
            format(self.exp_id, self.alpha, self.n_split)
        self.dataset = config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        data_cache_dir = os.path.join(cache_dir, self.dataset)
        self.min_freq=config.get('min_freq',1)
        self.vocab_path = data_cache_dir+'/vocab_{}_True_{}.pkl'.format(self.dataset, self.min_freq)
        self.vocab = WordVocab.load_vocab(self.vocab_path)

    def run(self, model, label):
        self._logger.info("-------- Speed Inference --------")
        x=model.get_static_embedding()

        # change the y same to the x
        y = np.array(label['speed']['speed']) #shape=5269
        index=np.array(label['speed']['index'])
        index_x=[]
        index_y=[]
        for i in range(index.shape[0]):
            geo_uid=index[i]
            if geo_uid in self.vocab.loc2index:
                index_y.append(i)
                index_x.append(self.vocab.loc2index[geo_uid])
        
                
        x=x[index_x]
        y=y[index_y]
        kf = KFold(n_splits=self.n_split)
        y_preds = []
        y_truths = []

        for train_index, test_index in kf.split(x):
            train_index = list(train_index)
            test_index = list(test_index)

            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            
            if choice <100:
                cur_lens=len(X_train)
                k=int(cur_lens*choice/100)
                index_range=list(range(cur_lens))
                select_index=random.choices(index_range,k=k)
                X_train=X_train[select_index]
                y_train=y_train[select_index]
                if i==0:
                    print(f"do choice with {choice}%")
                    print(f"befor is {cur_lens}")
                    after=len(X_train)
                    print(f"after is {after}")
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
        # np.save(save_path, predict)
        self._logger.info('Speed Inference Model is saved at {}'.format(save_path))

