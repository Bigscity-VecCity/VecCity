import numpy as np
import torch

from veccity.executor.general_executor import GeneralExecutor


class LINEExecutor(GeneralExecutor):
    def __init__(self, config, model, data_feature):
        GeneralExecutor.__init__(self, config, model, data_feature)
        self.loss_func = None

    def evaluate(self, test_dataloader):
        """
        use model to test data
        """
        self.evaluator.evaluate()

        with torch.no_grad():
            self.model.eval()
            # TODO 处理自定义 lossfunc
            loss_func = self.model.calculate_loss
            losses = []
            for batch in test_dataloader:
                batch.to_tensor(self.device)
                loss = loss_func(batch)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            return mean_loss
