import numpy as np
from logging import getLogger

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from veccity.utils import geojson2geometry
from veccity.downstream.downstream_models.abstract_model import AbstractModel


class KmeansModel(AbstractModel):

    def __init__(self, config):
        self._logger = getLogger()
        self.n_clusters = config.get('n_clusters', 2)
        self.representation_object = config.get('representation_object','region')
        self.random_state = config.get('random_state',3)
        self.exp_id = config.get('exp_id', None)
        self.dataset = config.get('dataset','')
        self.model = config.get('model','')
        self.output_dim = config.get('output_dim', 96)
        self.result_path = './veccity/cache/{}/evaluate_cache/kmeans_category_{}_{}.json'.\
            format(self.exp_id,self.n_clusters,self.random_state)
        self.qgis_result_path = './veccity/cache/{}/evaluate_cache/kmeans_qgis_{}_{}_{}_{}.csv'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim, self.n_clusters)
        self.data_path = './raw_data/' + self.dataset + '/'
        self.geo_file = config.get('geo_file', self.dataset)

    def run(self,node_emb,label):
        self.n_clusters = np.unique(label).shape[0]
        kmeans = KMeans(n_clusters=self.n_clusters,random_state=self.random_state)
        self._logger.info("K-Means Cluster:n_clusters={},random_state={}".format(self.n_clusters,self.random_state))
        predict = kmeans.fit_predict(node_emb)
        np.save(self.result_path,predict)
        if self.representation_object == 'region':
            self.region_cluster_visualize(predict)
        nmi = normalized_mutual_info_score(label, predict)
        ars = adjusted_rand_score(label, predict)
        result={'nmi':nmi,'ars':ars}
        self._logger.info("finish Kmeans cluster,result is {nmi="+str(nmi)+",ars="+str(ars)+"}")
        return result
    
    def clear(self):
        pass
    
    def save_result(self, result_token,save_path, filename=None):
        pass

    def _load_geo(self):
        """
                加载.geo文件，格式[geo_id, type, coordinates, function,traffic_type]
        """
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        geofile_region = geofile[geofile['traffic_type'] == 'region']
        return geofile_region
    
    def region_cluster_visualize(self,y_pred):
        #QGIS可视化
        geofile = self._load_geo()
        df = []
        region_geometry = [geojson2geometry(coordinate) for coordinate in geofile['coordinates']]
        for i in range(len(y_pred)):
            df.append([i,y_pred[i],region_geometry[i]])
        df = pd.DataFrame(df)
        df.columns = ['region_id', 'class', 'wkt']
        df = df.sort_values(by='class')
        df.to_csv(self.qgis_result_path, index=False)
        self._logger.info('Kmeans result for QGIS is saved at {}'.format(self.qgis_result_path))

