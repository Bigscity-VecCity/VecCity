import torch.nn as nn


class AbstractReprLearningModel(nn.Module):

    def __init__(self, config, data_feature):
        nn.Module.__init__(self)
        self.config = config
        self.data_feature = data_feature
        self.optimizer = None

    def run(self, train_dataloader=None,eval_dataloader=None):
        """
        Args:
            data : input of tradition model

        Returns:
            output of tradition model
        """

    def save_model(self, cache_name):
        """
        Args:
            cache_name : path to save parameters
        """
    
    def load_model(self, cache_name):
        """
        Args:
            cache_name : path to load parameters
        """
    
    def get_representation(self):
        """
        Returns:
            node embedding or model parameters
        """
    