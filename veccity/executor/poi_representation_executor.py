import torch
import numpy as np
import os
from logging import getLogger
from torch.utils.tensorboard import SummaryWriter

from veccity.executor.abstract_executor import AbstractExecutor
from veccity.utils import get_evaluator, ensure_dir
from veccity.upstream.poi_representation.static import StaticEmbed, DownstreamEmbed


class POIRepresentationExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        self._logger = getLogger()
        self.config = config
        self.data_feature = data_feature
        self.device = self.config.get('device', torch.device('cpu'))
        self._logger.info(self.device)
        self.model = model.to(self.device)
        self.exp_id = self.config.get('exp_id', None)

        self.cache_dir = './veccity/cache/{}/model_cache'.format(self.exp_id)
        self.evaluate_res_dir = './veccity/cache/{}/evaluate_cache'.format(self.exp_id)
        self.summary_writer_dir = './veccity/cache/{}/'.format(self.exp_id)
        ensure_dir(self.cache_dir)
        ensure_dir(self.evaluate_res_dir)
        ensure_dir(self.summary_writer_dir)
        self._writer = SummaryWriter(self.summary_writer_dir)
        # self._scaler = self.data_feature.get('scaler')
        self._logger.info(self.model)
        for name, param in self.model.named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))
        total_num = sum([param.nelement() for param in self.model.parameters()])
        self._logger.info('Total parameter numbers: {}'.format(total_num))

        self.embed_epoch = self.config.get('embed_epoch', 128)
        self.is_static = self.config.get('is_static', False)
        self.learner = self.config.get('learner', 'adam')
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.weight_decay = self.config.get('weight_decay', 0)
        self.lr_beta1 = self.config.get('lr_beta1', 0.9)
        self.lr_beta2 = self.config.get('lr_beta2', 0.999)
        self.lr_betas = (self.lr_beta1, self.lr_beta2)
        self.lr_alpha = self.config.get('lr_alpha', 0.99)
        self.lr_epsilon = self.config.get('lr_epsilon', 1e-8)
        self.lr_momentum = self.config.get('lr_momentum', 0)
        self.lr_decay = self.config.get('lr_decay', False)
        self.lr_scheduler_type = self.config.get('lr_scheduler', 'multisteplr')
        self.lr_decay_ratio = self.config.get('lr_decay_ratio', 0.1)
        self.milestones = self.config.get('steps', [])
        self.step_size = self.config.get('step_size', 10)
        self.lr_lambda = self.config.get('lr_lambda', lambda x: x)
        self.lr_T_max = self.config.get('lr_T_max', 30)
        self.lr_eta_min = self.config.get('lr_eta_min', 0)
        self.lr_patience = self.config.get('lr_patience', 10)
        self.lr_threshold = self.config.get('lr_threshold', 1e-4)
        self.clip_grad_norm = self.config.get('clip_grad_norm', False)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.)
        self.use_early_stop = self.config.get('use_early_stop', False)
        self.patience = self.config.get('patience', 50)
        self.log_every = self.config.get('log_every', 1)
        self.batch_size = self.config.get('batch_size', 64)
        self.w2v_window_size = self.config.get('w2v_window_size', 1)

        self.model_name = config.get('model')

        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

    def _build_optimizer(self):
        """
        根据全局参数`learner`选择optimizer
        """
        self._logger.info('You select `{}` optimizer.'.format(self.learner.lower()))
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                        momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate,
                                            eps=self.lr_epsilon, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate,
                                            alpha=self.lr_alpha, eps=self.lr_epsilon,
                                            momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.learning_rate,
                                               eps=self.lr_epsilon, betas=self.lr_betas)
        elif self.learner.lower() == 'asgd':
            optimizer = torch.optim.ASGD(self.model.parameters(), lr=self.learning_rate,
                                         alpha=self.lr_alpha, weight_decay=self.weight_decay)
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, weight_decay=self.weight_decay)
        return optimizer

    def _build_lr_scheduler(self):
        """
        根据全局参数`lr_scheduler`选择对应的lr_scheduler
        """
        if self.lr_decay:
            self._logger.info('You select `{}` lr_scheduler.'.format(self.lr_scheduler_type.lower()))
            if self.lr_scheduler_type.lower() == 'multisteplr':
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=self.milestones, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'steplr':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=self.step_size, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'exponentiallr':
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'cosineannealinglr':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.lr_T_max, eta_min=self.lr_eta_min)
            elif self.lr_scheduler_type.lower() == 'lambdalr':
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=self.lr_lambda)
            elif self.lr_scheduler_type.lower() == 'reducelronplateau':
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', patience=self.lr_patience,
                    factor=self.lr_decay_ratio, threshold=self.lr_threshold)
            else:
                self._logger.warning('Received unrecognized lr_scheduler, '
                                     'please check the parameter `lr_scheduler`.')
                lr_scheduler = None
        else:
            lr_scheduler = None
        return lr_scheduler

    def train(self, train_dataloader, eval_dataloader):
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '.m'
        if not self.config.get('train') and os.path.exists(model_path):
            self.load_model(model_path)
            return
        self._logger.info('Start training ...')
        if self.model_name == 'DownstreamEmbed':
            embed_layer = DownstreamEmbed(self.config, self.data_feature)
            self.data_feature['embed_layer'] = embed_layer
            return

        for epoch in range(self.embed_epoch):
            losses = []
            for batch in train_dataloader.next_batch():
                loss = self.model.calculate_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                losses.append(loss.item())

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    val_loss = loss.detach().cpu().numpy().tolist()
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()
        

            self._writer.add_scalar('training loss', np.mean(losses), epoch)
            self._logger.info("epoch {} complete! training loss is {:.2f}.".format(epoch, np.mean(losses)))

            if self.is_static:
                embed_mat = self.model.static_embed()
                embed_layer = StaticEmbed(embed_mat)
                self.data_feature['embed_layer'] = embed_layer
            else:
                self.data_feature['embed_layer'] = self.model

    def evaluate(self, test_dataloader):
        """
        use model to test data
        """
        evaluator = get_evaluator(self.config, self.data_feature)
        evaluator.evaluate()

    def load_model(self, cache_name):
        self._logger.info("Loaded model at " + cache_name)
        model_state, optimizer_state = torch.load(cache_name,map_location=torch.device("cpu"))
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        if self.is_static:
            if self.config.get('model') == 'DownstreamEmbed':
                embed_layer = DownstreamEmbed(self.config, self.data_feature)
            else:
                embed_mat = self.model.static_embed()
                embed_layer = StaticEmbed(embed_mat)
            self.data_feature['embed_layer'] = embed_layer
        else:
            self.data_feature['embed_layer'] = self.model

    def save_model(self, model_cache_file):
        ensure_dir(self.cache_dir)
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '.m'
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), model_path)
        self._logger.info("Saved model at {}".format(model_path))

        if self.is_static:
            if self.config.get('model') == 'DownstreamEmbed':
                embed_layer = DownstreamEmbed(self.config, self.data_feature)
            else:
                embed_mat = self.model.static_embed()
                embed_layer = StaticEmbed(embed_mat)
            self.data_feature['embed_layer'] = embed_layer
        else:
            self.data_feature['embed_layer'] = self.model
