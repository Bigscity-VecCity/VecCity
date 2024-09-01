import torch
import torch.nn as nn
from logging import getLogger
from sklearn.metrics import  mean_squared_error, mean_absolute_error
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter

from veccity.evaluator.downstream_models.abstract_model import AbstractModel
from torch.utils.data import Dataset, DataLoader
from veccity.evaluator.utils import StandardScaler
from veccity.data.dataset.dataset_subclass.eta_dataset import ETADataset
import copy

class TrajEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, embedding, device):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding = embedding
        self.n_layers = n_layers
        self.device = device
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=0.1 if n_layers > 1 else 0.0, batch_first=True)

    def forward(self, path, valid_len):
        
        original_shape = path.shape  # [batch_size, traj_len]
        
        full_embed = [torch.from_numpy(self.embedding[int(i)]).to(torch.float32) for i in path.view(-1)]
        full_embed = torch.stack(full_embed)
        full_embed = full_embed.view(*original_shape, self.input_dim)  # [batch_size, traj_len, embed_size]
        pack_x = pack_padded_sequence(full_embed, lengths=valid_len, batch_first=True, enforce_sorted=False).to(self.device)
        h0 = torch.zeros(self.n_layers, full_embed.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_layers, full_embed.size(0), self.hidden_dim).to(self.device)
        out, _ = self.lstm(pack_x, (h0, c0))
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = torch.stack([out[i, int(ind - 1), :] for i, ind in enumerate(valid_len)])  # [batch_size, hidden_dim]
        return out


class MLPReg(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, activation, embedding, is_static,device, max_len):
        super(MLPReg, self).__init__()

        self.num_layers = num_layers
        self.activation = activation   
        self.encoder=embedding.encode_sequence
        
        self.device = device
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.layers = []
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(self.num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, batch):
       
        x=self.encoder(batch)

        for i in range(self.num_layers - 1):
            x = self.activation(self.layers[i](x))
        return self.layers[-1](x).squeeze(1)


class TimeEstimationDataset(Dataset):
    def __init__(self, data_x,data_lens, data_y):
        self.x = data_x
        self.lens = data_lens
        self.y = data_y
        assert len(self.x) == len(self.y)

    def __getitem__(self, item):
        return self.x[item],self.lens[item], self.y[item]

    def __len__(self):
        return len(self.x)

class TravelTimeEstimationModel(AbstractModel):
    def __init__(self, config):
        self.config=config
        self._logger = getLogger()
        self.alpha = config.get('alpha', 1)
        self.n_split = config.get('n_split', 5)
        self.exp_id = config.get('exp_id', None)
        self.result_path = './veccity/cache/{}/evaluate_cache/regression_{}_{}.npy'. \
            format(self.exp_id, self.alpha, self.n_split)
        eta_dataset=ETADataset(config)
        self.device=config.get('device')
        self.train_dataloader,self.eval_dataloader,self.test_dataloader = eta_dataset.get_data()
        self.summary_writer_dir = './veccity/cache/{}'.format(self.exp_id)
        self._writer = SummaryWriter(self.summary_writer_dir)


    def run(self, embed_model,**kwargs):
        input_dim=self.config.get('embed_size',128)
        hidden_dim=self.config.get('downstream_hidden_size',512)
        is_static=self.config.get('is_static',False)
        device=self.config.get("device")
        max_len=self.config.get("max_len",128)
        max_epoch=self.config.get("task_epoch",100)

        model = MLPReg(input_dim, hidden_dim, 2, nn.ReLU(),embed_model,is_static,device,max_len).to(device)
        opt = torch.optim.Adam(model.parameters(),lr=1e-4)
        loss_fn=nn.MSELoss()
        patience = 100

        best = {"best epoch": 0, "mae": 1e9, "rmse": 1e9}

        best_model=None
        self._logger.info("-------- ETA START --------")
        for epoch in range(max_epoch):
            model.train()
            losses=[]
            for batch in self.train_dataloader:
                opt.zero_grad()
                batch.update(kwargs)
                preds=model(batch)
                loss = loss_fn(preds, batch['targets'].squeeze().to(self.device))
                loss.backward()
                opt.step()
                losses.append(loss.item())
            
            self._writer.add_scalar('ETA Train loss', np.mean(losses), epoch)
            self._logger.info(f"ETA EPOCH {epoch} Train loss : {np.mean(losses)}")
                
            model.eval()
            y_preds = []
            y_trues = []
            for batch in self.eval_dataloader:
                y_preds.append(model(batch).detach().cpu())
                y_trues.append(batch['targets'].detach().cpu())
            
            y_preds = torch.cat(y_preds, dim=0)
            y_trues = torch.cat(y_trues, dim=0)

            mae = mean_absolute_error(y_trues, y_preds)
            rmse = mean_squared_error(y_trues, y_preds) ** 0.5
            self._logger.info(f'Epoch: {epoch}, MAE: {mae.item():.4f}, RMSE: {rmse.item():.4f}')
            self._writer.add_scalar('ETA Valid MAE', mae, epoch)
            self._writer.add_scalar('ETA Valid RMSE', rmse, epoch)
            if mae < best["mae"]:
                best = {"best epoch": epoch, "mae": mae, "rmse": rmse}
                patience = 10
                best_model=copy.deepcopy(model)
            else:
                patience -= 1
                if not patience:
                    self._logger.info("Best epoch: {}, MAE:{}, RMSE:{}".format(best['best epoch'], best['mae'], best["rmse"]))
                    break

        model=best_model
        model.eval()
        y_preds = []
        y_trues = []
        for batch in self.test_dataloader:
            y_preds.append(model(batch).detach().cpu())
            y_trues.append(batch['targets'].detach().cpu())
        
        y_preds = torch.cat(y_preds, dim=0)
        y_trues = torch.cat(y_trues, dim=0)
        mae = mean_absolute_error(y_trues, y_preds)
        rmse = mean_squared_error(y_trues, y_preds) ** 0.5
        best['mae']=mae
        best['rmse']=rmse
        self._logger.info("Test result:epoch {}, MAE:{}, RMSE:{}".format(best['best epoch'], best['mae'], best["rmse"]))
        self._writer.close()
        return best

    def clear(self):
        pass

    def save_result(self, save_path, filename=None):
        pass