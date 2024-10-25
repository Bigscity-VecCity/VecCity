from veccity.utils import get_evaluator, ensure_dir
from veccity.executor.abstract_executor import AbstractExecutor
import torch
from logging import getLogger


class TwoStepExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        self.evaluator = get_evaluator(config,data_feature)
        self.config = config
        self.model = model
        self.exp_id = config.get('exp_id', None)
        self._logger = getLogger()

        # total_num = sum([param.nelement() for param in self.model.parameters()])
        # self._logger.info('Total parameter numbers: {}'.format(total_num))

        self.cache_dir = './veccity/cache/{}/model_cache'.format(self.exp_id)
        self.evaluate_res_dir = './veccity/cache/{}/evaluate_cache'.format(self.exp_id)
        ensure_dir(self.cache_dir)
        ensure_dir(self.evaluate_res_dir)

    def evaluate(self, test_dataloader=None):
        """
        use model to test data
        """
        self.evaluator.evaluate(model=self.model)
        test_result = self.evaluator.save_result(self.evaluate_res_dir)
        return test_result

    def train(self, train_dataloader=None, eval_dataloader=None):
        """
        use data to train model with config
        """
        return self.model.run(train_dataloader,eval_dataloader)


    def load_model(self, cache_name):
        self._logger.info("Loaded model at " + cache_name)
        model_state  = torch.load(cache_name,map_location=torch.device("cpu"))
        try:
            self.model.load_state_dict(model_state['model_state_dict'])
            if 'optimizer_state_dict' in model_state:
                self.model.optimizer.load_state_dict(model_state['optimizer_state_dict'])
        except:
            pass

    def save_model(self, epoch):
        ensure_dir(self.cache_dir)
        config = dict()
        config['model_state_dict'] = self.model.state_dict()
        if self.model.optimizer != None:
            config['optimizer_state_dict'] = self.model.optimizer.state_dict() 
        config['epoch'] = epoch
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '.m' 
        torch.save(config, model_path)
        self._logger.info("Saved model at {}".format(epoch))
        return model_path
