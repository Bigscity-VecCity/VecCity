import numpy as np
from veccity.executor.general_executor import GeneralExecutor

import torch


class JGRMExecutor(GeneralExecutor):
    def __init__(self, config, model, data_feature):
        GeneralExecutor.__init__(self, config, model, data_feature)

    def _train_epoch(self, train_dataloader, epoch_idx, loss_func=None):
        """
        完成模型一个轮次的训练

        Args:
            train_dataloader: 训练数据
            epoch_idx: 轮次数
            loss_func: 损失函数

        Returns:
            list: 每个batch的损失的数组
        """
        self.model.train()
        loss_func = loss_func if loss_func is not None else self.model.calculate_loss
        losses = []
        for batch in train_dataloader:
            self.optimizer.zero_grad()
            gps_data, gps_assign_mat, route_data, route_assign_mat, gps_length = batch
            gps_data = gps_data.to(self.device)
            gps_assign_mat = gps_assign_mat.to(self.device)
            route_data = route_data.to(self.device)
            route_assign_mat = route_assign_mat.to(self.device)
            gps_length = gps_length.to(self.device)
            batch = (gps_data, gps_assign_mat, route_data, route_assign_mat, gps_length)
            # batch.to_tensor(self.device)
            loss = loss_func(batch)
            self._logger.debug(loss.item())
            losses.append(loss.item())
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.optimizer.step()
        return losses

    def _valid_epoch(self, eval_dataloader, epoch_idx, loss_func=None):
        """
        完成模型一个轮次的评估

        Args:
            eval_dataloader: 评估数据
            epoch_idx: 轮次数
            loss_func: 损失函数

        Returns:
            float: 评估数据的平均损失值
        """
        with torch.no_grad():
            self.model.eval()
            loss_func = (
                loss_func if loss_func is not None else self.model.calculate_loss
            )
            losses = []
            for batch in eval_dataloader:
                gps_data, gps_assign_mat, route_data, route_assign_mat, gps_length = (
                    batch
                )
                gps_data = gps_data.to(self.device)
                gps_assign_mat = gps_assign_mat.to(self.device)
                route_data = route_data.to(self.device)
                route_assign_mat = route_assign_mat.to(self.device)
                gps_length = gps_length.to(self.device)
                batch = (
                    gps_data,
                    gps_assign_mat,
                    route_data,
                    route_assign_mat,
                    gps_length,
                )
                loss = loss_func(batch)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            self._writer.add_scalar("eval loss", mean_loss, epoch_idx)
            return mean_loss
