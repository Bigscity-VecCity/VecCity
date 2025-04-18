import csv
import numpy as np
from logging import getLogger
from veccity.downstream.abstract_evaluator import AbstractEvaluator
from veccity.downstream.downstream_models.loc_pred_model import *
from sklearn.model_selection import  StratifiedKFold
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score
from veccity.upstream.poi_representation.static import StaticEmbed, DownstreamEmbed
from sklearn.metrics import normalized_mutual_info_score
from veccity.upstream.poi_representation.utils import next_batch
import copy
import random



class POIRepresentationEvaluator(AbstractEvaluator):

    def __init__(self, config, data_feature):
        self._logger = getLogger()
        self.config = config
        self.data_feature = data_feature
        self.model_name = self.config.get('downstream_model', 'gru')
        self.device = self.config.get('device')
        self.result = {}
        self.model = config.get('model')
        self.dataset = config.get('dataset')
        self.exp_id = config.get('exp_id')
        self.embed_size = config.get('embed_size')
        self.hidden_size = config.get("downstream_hidden_size",512)
        self.num_loc = self.data_feature.get('num_loc')
        self.embed_layer = self.data_feature.get('embed_layer')
        self.task_epoch = self.config.get('task_epoch', 5)
        self.batch_size = self.config.get('downstream_batch_size', 32)
        self.choice=self.config.get('choice',0)

    def collect(self, batch):
        pass

    def evaluate_loc_pre(self):
        st_aux_embed_size = self.config.get('st_aux_embed_size', 32)
        train_set = self.data_feature.get('train_set')
        choice=self.config.get('choice',0)
        if choice:
            train_set=random.choices(train_set,k=choice)
            self._logger.info(f'select train size {choice} in lp')

        test_set = self.data_feature.get('test_set')
        embed_layer=copy.deepcopy(self.embed_layer)       
            
        self.result['loc_pre_acc1'], self.result['loc_pre_acc5'], self.result['loc_pre_f1_macro'] =\
        trajectory_based_classification(train_set, test_set, self.num_loc,embed_layer=embed_layer,embed_size=self.embed_size,hidden_size=self.hidden_size,
                       num_epoch=self.task_epoch, num_loc=self.num_loc,batch_size=self.batch_size,aux_embed_size=st_aux_embed_size, task="LP",device=self.device)
    
    def evaluate_traj_clf(self):
        num_user = self.data_feature.get('num_user')
        st_aux_embed_size = self.config.get('st_aux_embed_size', 32)
        train_set = self.data_feature.get('train_set')
        test_set = self.data_feature.get('test_set')
        embed_layer = copy.deepcopy(self.embed_layer)

        choice=self.config.get('choice',0)
        if choice:
            train_set=random.choices(train_set,k=choice)
            self._logger.info(f'select train size {choice} in tul')

        self.result['traj_clf_acc1'],self.result['traj_clf_acc5'], self.result['traj_clf_f1_macro'] =\
        trajectory_based_classification(train_set, test_set, num_class=num_user,embed_layer=embed_layer,embed_size=self.embed_size,hidden_size=self.hidden_size,
                       num_epoch=self.task_epoch, num_loc=self.num_loc,batch_size=self.batch_size,aux_embed_size=st_aux_embed_size, task="TUL",device=self.device)
        
    
    def evaluate_loc_clf(self):
        logger=getLogger()
        logger.info('Start training downstream model [loc_clf]...')
        
        if not self.config.get('is_static', True):
            embed_layer=self.embed_layer.static_embed()
            embed_layer=StaticEmbed(embed_layer)
        else:
            embed_layer=self.embed_layer

        # build model
        num_loc = self.data_feature.get('num_loc')
        seed = self.config.get('seed',31)
        embed_size = self.config.get('embed_size', 128)
        task_epoch = self.config.get('task_epoch', 5)
        category = self.data_feature.get('coor_df')
        device=self.config.get('device','cuda:0')
        downstream_batch_size = self.data_feature.get('downstream_batch_size', 32)
        # 随机划分数据集
        assert num_loc == len(category)
        inputs=category.geo_uid.to_numpy()
        labels=category.category.to_numpy()
        
        num_class = labels.max()+1
        
        skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
        score_log = []
        for i,(train_ind,valid_ind) in enumerate(skf.split(inputs,labels)):
            clf_model = FCClassifier(copy.deepcopy(embed_layer),embed_size,num_class,hidden_size=128).to(device)
            optimizer = torch.optim.Adam(clf_model.parameters(), lr=1e-4)
            loss_func = nn.CrossEntropyLoss()
            best_loss=100
            best_model=None
            for epoch in range(task_epoch):
                losses=[]
                for _, batch in enumerate(next_batch(train_ind, downstream_batch_size)):
                    bacth_input = torch.tensor(inputs[batch],dtype=torch.long,device=device)
                    batch_label = torch.tensor(labels[batch],dtype=torch.long,device=device)
                    out=clf_model(bacth_input)
                    loss=loss_func(out,batch_label)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                if np.mean(losses) < best_loss:
                    best_model=copy.deepcopy(clf_model)
            
            clf_model=best_model

            pres_raw=[]
            test_labels=[]
            for _, batch in enumerate(next_batch(valid_ind, downstream_batch_size)):
                bacth_input = torch.tensor(inputs[batch],dtype=torch.long,device=device)
                batch_label = torch.tensor(labels[batch],dtype=torch.long,device=device)
                out=clf_model(bacth_input)

                pres_raw.append(out.detach().cpu().numpy())
                test_labels.append(batch_label.detach().cpu().numpy())

            pres_raw, test_labels = np.concatenate(pres_raw), np.concatenate(test_labels)
            pres = pres_raw.argmax(-1)

            
            acc = accuracy_score(test_labels, pres)
            f1_macro = f1_score(test_labels, pres, average='macro')
            score_log.append([acc, f1_macro])
            
        best_acc,best_f1_macro = np.mean(score_log, axis=0)
        logger.info('Acc %.6f, F1-macro %.6f' % (
            best_acc, best_f1_macro))
        self.result['loc_clf_acc'] = best_acc
        self.result['loc_clf_f1_macro'] = best_f1_macro
        # 记录结果

    def evaluate_loc_cluster(self):
        embed_layer = self.data_feature.get('embed_layer')

        if not self.config.get('is_static', True):
            embed_layer=embed_layer.static_embed()
            embed_layer=StaticEmbed(embed_layer)
        
        category = self.data_feature.get('coor_df')

        inputs=category.geo_uid.to_numpy()
        labels=category.category.to_numpy()
        num_class = labels.max()+1

        node_embedding=embed_layer(torch.tensor(inputs)).cpu()
        self.evaluation_cluster(labels,node_embedding,num_class)
        

    def evaluation_cluster(self, y_truth, node_emb, kinds):
        self._logger.info('Start Kmeans, data.shape = {}, kinds = {}'.format(
            str(node_emb.shape), kinds))
        k_means = KMeans(n_clusters=kinds, random_state=2024)
        k_means.fit(node_emb)
        labels = k_means.labels_
        y_predict = k_means.predict(node_emb)
        y_predict_useful = y_predict
        nmi = normalized_mutual_info_score(y_truth, y_predict_useful)
        ars = adjusted_rand_score(y_truth, y_predict_useful)
        # SC指数
        sc = float(metrics.silhouette_score(node_emb, labels, metric='euclidean'))
        # DB指数
        db = float(metrics.davies_bouldin_score(node_emb, labels))
        # CH指数
        ch = float(metrics.calinski_harabasz_score(node_emb, labels))
        self._logger.info("Evaluate result [loc_cluaster] is sc = {:6f}, db = {:6f}, ch = {:6f}, nmi = {:6f}, ars = {:6f}".format(sc, db, ch, nmi, ars))
        self.result['sc'] = sc
        self.result['db'] = db
        self.result['ch'] = ch
        self.result['nmi'] = nmi
        self.result['ars'] = ars
        return sc, db, ch, nmi, ars

    def evaluate(self):
        self._logger.info('Start evaluating ...')
        self.evaluate_loc_pre()
        self.evaluate_traj_clf()
        poi_type_name = self.config.get('poi_type_name', None)
        if poi_type_name is not None:
            self.evaluate_loc_clf()
            # self.evaluate_loc_cluster()
        result_path = './veccity/cache/{}/evaluate_cache/{}_evaluate_{}_{}_{}_{}.json'. \
            format(self.exp_id, self.exp_id, self.model, self.dataset, str(self.embed_size),str(self.choice))
        self._logger.info(self.result)

        def dict_to_csv(dictionary, filename):
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=dictionary.keys())
                writer.writeheader()
                writer.writerow(dictionary)
        result_path = './veccity/cache/{}/evaluate_cache/{}_evaluate_{}_{}_{}_{}.csv'. \
            format(self.exp_id, self.exp_id, self.model, self.dataset, str(self.embed_size),str(self.choice))
        dict_to_csv(self.result, result_path)
        self._logger.info('Evaluate result is saved at {}'.format(result_path))

    def save_result(self, save_path, filename=None):
        pass

    def clear(self):
        pass


class FCClassifier(nn.Module):
    def __init__(self, embed_layer, input_size, output_size, hidden_size):
        super().__init__()

        self.embed_layer = embed_layer
        self.add_module('embed_layer', self.embed_layer)

        self.input_linear = nn.Linear(input_size, hidden_size)
        self.hidden_linear = nn.Linear(hidden_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, output_size)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        :param x: input batch of location tokens, shape (batch_size)
        :return: prediction of the corresponding location categories, shape (batch_size, output_size)
        """
        h = self.dropout(self.embed_layer(x))  # (batch_size, input_size)
        h = self.dropout(self.act(self.input_linear(h)))
        h = self.dropout(self.act(self.hidden_linear(h)))
        out = self.output_linear(h)
        return out