import os
import torch
from logging import getLogger
from torch.utils.tensorboard import SummaryWriter
from veccity.utils import ensure_dir
from veccity.executor.scheduler import CosineLRScheduler
from veccity.utils import get_evaluator
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import numpy as np


class LineSeqExecutor(object):

    def __init__(self, config, model, data_feature):
        self.config = config
        self.data_feature = data_feature
        self._logger = getLogger()
        self.model=config.get('model')
        self.dataset=config.get('dataset')

        self.vocab_size = self.data_feature.get('vocab_size')
        self.usr_num = self.data_feature.get('usr_num')

        self.exp_id = self.config.get('exp_id', None)
        self.device = self.config.get('device', torch.device('cpu'))
        self.epochs = self.config.get('max_epoch', 100)
        self.model_name = self.config.get('model', '')
        self.embedding_size=self.config.get('emebed_size',128)

        # optimizer
        self.learner = self.config.get('learner', 'adamw')
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.weight_decay = self.config.get('weight_decay', 0.01)
        self.lr_beta1 = self.config.get('lr_beta1', 0.9)
        self.lr_beta2 = self.config.get('lr_beta2', 0.999)
        self.lr_betas = (self.lr_beta1, self.lr_beta2)
        self.lr_alpha = self.config.get('lr_alpha', 0.99)
        self.lr_epsilon = self.config.get('lr_epsilon', 1e-8)
        self.lr_momentum = self.config.get('lr_momentum', 0)
        self.grad_accmu_steps = self.config.get('grad_accmu_steps', 1)
        self.test_every = self.config.get('test_every', 200)

        self.lr_decay = self.config.get('lr_decay', True)
        self.lr_scheduler_type = self.config.get('lr_scheduler', 'cosinelr')
        self.lr_decay_ratio = self.config.get('lr_decay_ratio', 0.1)
        self.milestones = self.config.get('steps', [])
        self.step_size = self.config.get('step_size', 10)
        self.lr_lambda = self.config.get('lr_lambda', lambda x: x)
        self.lr_T_max = self.config.get('lr_T_max', 30)
        self.lr_eta_min = self.config.get('lr_eta_min', 0)
        self.lr_patience = self.config.get('lr_patience', 10)
        self.lr_threshold = self.config.get('lr_threshold', 1e-4)
        self.lr_warmup_epoch = self.config.get("lr_warmup_epoch", 0)
        self.lr_warmup_init = self.config.get("lr_warmup_init", 1e-6)
        self.t_in_epochs = self.config.get("t_in_epochs", True)

        self.clip_grad_norm = self.config.get('clip_grad_norm', False)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.)
        self.use_early_stop = self.config.get('use_early_stop', False)
        self.patience = self.config.get('patience', 50)
        self.log_every = self.config.get('log_every', 1)
        self.log_batch = self.config.get('log_batch', 10)
        self.saved = self.config.get('saved_model', True)
        self.load_best_epoch = self.config.get('load_best_epoch', True)
        self.l2_reg = self.config.get('l2_reg', None)

        self.road_embedding_path = './veccity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.embedding_size)
        self.cache_dir = './veccity/cache/{}/model_cache'.format(self.exp_id)
        self.png_dir = './veccity/cache/{}'.format(self.exp_id)
        self.evaluate_res_dir = './veccity/cache/{}/evaluate_cache'.format(self.exp_id)
        self.summary_writer_dir = './veccity/cache/{}'.format(self.exp_id)
        ensure_dir(self.cache_dir)
        ensure_dir(self.png_dir)
        ensure_dir(self.evaluate_res_dir)
        ensure_dir(self.summary_writer_dir)
        self._writer = SummaryWriter(self.summary_writer_dir)

        self.model = model.to(self.device)  # bertlm
        
        self._logger.info(self.model)
        for name, param in self.model.named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))
        total_num = sum([param.nelement() for param in self.model.parameters()])
        self._logger.info('Total parameter numbers: {}'.format(total_num))


        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()
        self.optimizer.zero_grad()
        # self.load_model_with_epoch(28)
        # self.model.to(self.device)
        self.evaluator = get_evaluator(self.config, self.data_feature)


    def save_model(self, cache_name):
        """
        将当前的模型保存到文件

        Args:
            cache_name(str): 保存的文件名
        """
        ensure_dir(self.cache_dir)
        config = dict()
        # config['model'] = self.model.cpu()
        config['model_state_dict']=self.model.state_dict()
        config['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(config, cache_name)
        self.model.to(self.device)
        self._logger.info("Saved model at " + cache_name)

    def load_model_state(self, cache_name):
        """
        加载对应模型的 cache （用于加载参数直接进行测试的场景）

        Args:
            cache_name(str): 保存的文件名
        """
        assert os.path.exists(cache_name), 'Weights at {} not found' % cache_name
        checkpoint = torch.load(cache_name, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'].state_dict())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._logger.info("Loaded model at " + cache_name)

    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        assert os.path.exists(cache_name), 'Weights at {} not found' % cache_name

        checkpoint = torch.load(cache_name, map_location='cpu')
        # try:
        # import pdb
        # pdb.set_trace()
        # checkpoint['model_state_dict']['model.transformer.embed.tok_embed.weight']=checkpoint['model_state_dict']['model.transformer.embed.tok_embed.weight'][:self.model.model.transformer.embed.tok_embed.weight.shape[0]]

            # checkpoint['model_state_dict']['model.node_embedding.weight']=checkpoint['model_state_dict']['model.node_embedding.weight'][:self.model.model.node_embedding.weight.shape[0]]
        # checkpoint['model_state_dict']['model.decoder.weight'] = checkpoint['model_state_dict']['model.decoder.weight'][:self.model.model.transformer.embed.tok_embed.weight.shape[0]]
        # checkpoint['model_state_dict']['model.decoder.bias'] = checkpoint['model_state_dict']['model.decoder.bias'][:self.model.model.transformer.embed.tok_embed.weight.shape[0]]
        # except Exception:
        # checkpoint['model_state_dict']=checkpoint['model'].state_dict()
        # import pdb
        # pdb.set_trace()
        # checkpoint['model_state_dict']['model.transformer.embed.tok_embed.weight'][:self.model.model.transformer.embed.tok_embed.weight.shape[0]]=checkpoint['model_state_dict']['model.transformer.embed.tok_embed.weight'][:self.model.model.transformer.embed.tok_embed.weight.shape[0]]
        # checkpoint['model_state_dict']['model.decoder.weight'] = checkpoint['model_state_dict']['model.decoder.weight'][:self.model.model.transformer.embed.tok_embed.weight.shape[0]]
        # checkpoint['model_state_dict']['model.decoder.bias'] = checkpoint['model_state_dict']['model.decoder.bias'][:self.model.model.transformer.embed.tok_embed.weight.shape[0]]
        # try:
        #     checkpoint['model_state_dict']=checkpoint['model'].state_dict()
        #     checkpoint['model_state_dict']['model.node_embedding.weight'] = checkpoint['model_state_dict']['model.node_embedding.weight'][:self.model.model.node_embedding.weight.shape[0]]
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        # except Exception:
        # checkpoint['model_state_dict']['model.node_embedding.weight'] = checkpoint['model_state_dict']['model.node_embedding.weight'][:self.model.model.node_embedding.weight.shape[0]]
        #     self.model.load_state_dict(checkpoint['model_state_dict'])


        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.save_model(cache_name)
        self._logger.info("Loaded model at " + cache_name)

    def save_model_with_epoch(self, epoch):
        """
        保存某个epoch的模型

        Args:
            epoch(int): 轮数
        """
        ensure_dir(self.cache_dir)
        config = dict()
        config['model_state_dict'] = self.model.state_dict()
        # config['optimizer_state_dict'] = self.optimizer.state_dict()
        # config['epoch'] = epoch
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        torch.save(config, model_path)
        self.model.to(self.device)
        self._logger.info("Saved model at {}".format(epoch))
        return model_path

    def load_model_with_epoch(self, epoch):
        """
        加载某个epoch的模型

        Args:
            epoch(int): 轮数
        """
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        assert os.path.exists(model_path), 'Weights at epoch %d not found' % epoch
        checkpoint = torch.load(model_path, map_location=self.device)
        # self.model = checkpoint['model'].to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._logger.info("Loaded model at {}".format(epoch))

    def _build_optimizer(self):
        """
        根据全局参数`learner`选择optimizer
        """
        self._logger.info('You select `{}` optimizer.'.format(self.learner.lower()))
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
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
            elif self.lr_scheduler_type.lower() == 'cosinelr':
                lr_scheduler = CosineLRScheduler(
                    self.optimizer, t_initial=self.epochs, lr_min=self.lr_eta_min, decay_rate=self.lr_decay_ratio, t_in_epochs=self.t_in_epochs)
            else:
                self._logger.warning('Received unrecognized lr_scheduler, '
                                     'please check the parameter `lr_scheduler`.')
                lr_scheduler = None
        else:
            lr_scheduler = None
        return lr_scheduler
    
    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self.evaluator.evaluate(model=self.model)
        test_result = self.evaluator.save_result(self.evaluate_res_dir)
        return test_result

    def train(self, train_dataloader, eval_dataloader, test_dataloader=None):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = -1
        train_time = []
        eval_time = []
        train_loss = []
        eval_loss = []
        eval_acc = []
        lr_list = []

        num_batches = len(train_dataloader)
        self._logger.info("Num_batches: epochs={}, train={}, eval={}".format(self.epochs,num_batches, len(eval_dataloader)))

        for epoch_idx in range(self.epochs):
            start_time = time.time()
            train_avg_loss = self._train_epoch(train_dataloader, epoch_idx)
            t1 = time.time()
            train_time.append(t1 - start_time)
            train_loss.append(train_avg_loss)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            eval_avg_loss = self._valid_epoch(eval_dataloader, epoch_idx, mode='Eval')
            end_time = time.time()
            eval_time.append(end_time - t2)
            eval_loss.append(eval_avg_loss)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(eval_avg_loss)
                elif self.lr_scheduler_type.lower() == 'cosinelr':
                    self.lr_scheduler.step(epoch_idx + 1)
                else:
                    self.lr_scheduler.step()

            log_lr = self.optimizer.param_groups[0]['lr']
            lr_list.append(log_lr)
            if (epoch_idx % self.log_every) == 0:
                message = 'Epoch [{}/{}] ({})  train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'. \
                    format(epoch_idx, self.epochs, (epoch_idx + 1) * num_batches, train_avg_loss,
                           eval_avg_loss, log_lr, (end_time - start_time))
                self._logger.info(message)

            if eval_avg_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, eval_avg_loss, model_file_name))
                min_val_loss = eval_avg_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break

            if (epoch_idx + 1) % self.test_every == 0:
                self.evaluate(test_dataloader)

        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)

        self._draw_png([(train_loss, eval_loss, 'loss'), (lr_list, 'lr')])
        node_embedding=self.model.get_static_embedding()
        np.save(self.road_embedding_path,node_embedding)
        return min_val_loss

    def _draw_png(self, data):
        for data_iter in data:
            plt.figure()
            if len(data_iter) == 3:
                train_list, eval_list, name = data_iter
                x_index = np.arange((len(train_list)))
                plt.plot(x_index, train_list, 'r', label='train_{}'.format(name))
                plt.plot(x_index, eval_list, 'b', label='eval_{}'.format(name))
            else:
                data_list, name = data_iter
                x_index = np.arange((len(data_list)))
                plt.plot(x_index, data_list, 'r', label='{}'.format(name))
            plt.ylabel(name)
            plt.xlabel('epoch')
            plt.title(str(self.exp_id) + ': ' + str(self.model_name))
            plt.legend()
            path = self.png_dir + '/{}_{}.png'.format(self.exp_id, name)
            plt.savefig(path)
            self._logger.info('Save png at {}'.format(path))

    def _train_epoch(self, train_dataloader, epoch_idx):
        batches_seen = epoch_idx * len(train_dataloader)  # 总batch数
        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        for i, batch in tqdm(enumerate(train_dataloader), desc="Train epoch={}".format(
                epoch_idx), total=len(train_dataloader)):
            loss=self.model.calculate_loss(batch,batches_seen+i)
            loss.backward()

            if batches_seen % self.grad_accmu_steps == 0:
                self.optimizer.step()
                if self.lr_scheduler_type == 'cosinelr' and self.lr_scheduler is not None:
                    self.lr_scheduler.step_update(num_updates=batches_seen // self.grad_accmu_steps)
                self.optimizer.zero_grad()

            with torch.no_grad():
                epoch_loss += loss.item()  # add total loss of batch
            
            post_fix = {
                "mode": "Train",
                "epoch": epoch_idx,
                "iter": i,
                "lr": self.optimizer.param_groups[0]['lr'],
                "loss": loss.item()
            }
            if i % self.log_batch == 0:
                self._logger.info(str(post_fix))
        epoch_loss = epoch_loss / len(train_dataloader)
        return epoch_loss

    @torch.no_grad()
    def _valid_epoch(self, eval_dataloader, epoch_idx, mode='Eval'):
        self.model = self.model.eval()
        epoch_loss = 0  # total loss of epoch
        for i, batch in tqdm(enumerate(eval_dataloader), desc="Train epoch={}".format(
                epoch_idx), total=len(eval_dataloader)):
            loss=self.model.calculate_loss(batch,0)
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / len(eval_dataloader)
        return epoch_loss
    