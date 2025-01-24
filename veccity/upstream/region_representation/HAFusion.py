import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
import time
import torch.optim as optim
from veccity.upstream.abstract_replearning_model import AbstractReprLearningModel
from veccity.utils import need_train
import pdb

def _mob_loss(s_embeddings, t_embeddings, mob):
    inner_prod = torch.mm(s_embeddings, t_embeddings.T)
    softmax1 = nn.Softmax(dim=-1)
    phat = softmax1(inner_prod)
    loss = torch.sum(-torch.mul(mob, torch.log(phat + 0.0001)))
    inner_prod = torch.mm(t_embeddings, s_embeddings.T)
    softmax2 = nn.Softmax(dim=-1)
    phat = softmax2(inner_prod)
    loss += torch.sum(-torch.mul(torch.transpose(mob, 0, 1), torch.log(phat + 0.0001)))
    return loss


def _general_loss(embeddings, adj):
    inner_prod = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    loss = F.mse_loss(inner_prod, adj)
    return loss


class ModelLoss(nn.Module):
    def __init__(self):
        super(ModelLoss, self).__init__()

    def forward(self, out_s, out_t, mob_adj, out_p, poi_sim, out_l, land_sim):
        mob_loss = _mob_loss(out_s, out_t, mob_adj)
        poi_loss = _general_loss(out_p, poi_sim)
        land_loss = _general_loss(out_l, land_sim)
        loss = poi_loss + land_loss + mob_loss
        return loss

class HAFusion(AbstractReprLearningModel):
    def __init__(self,config,data_feature):
        super().__init__(config,data_feature)
        self._logger = getLogger()
        self.device = config.get('device')
        self.embedding_size = config.get('output_dim', 96)
        self.dataset = config.get('dataset', '')
        self.epochs = config.get('max_epoch', 1)
        self.model = config.get('model', '')
        self.exp_id = config.get('exp_id', None)
        self.learning_rate = config.get('learning_rate', 0.005)
        self.weight_decay = config.get('weight_decay', 5e-4)
        self.d_prime = config.get('d_prime', 64)
        self.d_m = config.get('d_m', 72)
        self.c = config.get('c', 32)
        self.mob_adj = data_feature.get('mob_adj')
        self.poi_sim = data_feature.get('poi_simi')
        self.mob_feature = data_feature.get('mob_dist')
        self.POI_feature = data_feature.get('poi_dist')
        self.region_num = data_feature.get('num_regions')
        self.POI_dim = data_feature.get('poi_dim')
        self.landUse_dim = 1
        self.npy_cache_file = './veccity/cache/{}/evaluate_cache/region_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.embedding_size)
        
        
    def run(self, train_dataloader=None, eval_dataloader=None):
        if not need_train(self.config):
            return
        start_time = time.time()
        model = HAFusion_Model(self.POI_dim, self.landUse_dim, self.region_num, self.embedding_size, self.d_prime, self.d_m, self.c, self.config).to(self.device)
        model_loss = ModelLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self._logger.info(model)
        self._logger.info("start training,lr={},weight_dacay={}".format(self.learning_rate,self.weight_decay))

        landUse_feature = np.zeros((self.region_num, 1))
        landUse_feature = landUse_feature[np.newaxis]
        landUse_feature = torch.Tensor(landUse_feature).to(self.device)
        land_sim = torch.zeros((self.region_num, self.region_num), device=self.device)
        input_features = [self.POI_feature, landUse_feature, self.mob_feature]

        for epoch in range(self.epochs):
            model.train()
            out_s, out_t, out_p, out_l = model(input_features)
            # pdb.set_trace()
            loss = model_loss(out_s, out_t, self.mob_adj, out_p, self.poi_sim, out_l, land_sim)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self._logger.info("Epoch {}, Loss {}".format(epoch, loss.item()))
    
        t1 = time.time()-start_time
        outs = model.out_feature()
        self._logger.info('cost time is '+str(t1/self.epochs))
        total_num = sum([param.nelement() for param in model.parameters()])
        total_num += outs.view(-1).shape[0]
        total_num += sum([param.nelement() for param in model_loss.parameters()])
        self._logger.info('Total parameter numbers: {}'.format(total_num))
        node_embedding = outs
        node_embedding = node_embedding.detach().cpu().numpy()
        np.save(self.npy_cache_file, node_embedding)
        self._logger.info('词向量和模型保存完成')
        self._logger.info('词向量维度：(' + str(len(node_embedding)) + ',' + str(len(node_embedding[0])) + ')')

class DeepFc(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepFc, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.Linear(input_dim * 2, output_dim),
            nn.LeakyReLU(negative_slope=0.3, inplace=True), )

        self.output = None

    def forward(self, x):
        output = self.model(x)
        self.output = output
        return output

    def out_feature(self):
        return self.output


class RegionFusionBlock(nn.Module):

    def __init__(self, input_dim, nhead, dropout, dim_feedforward=2048):
        super(RegionFusionBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, nhead, dropout=dropout, batch_first=True, bias=True)
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(input_dim, dim_feedforward, )
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src):
        src2, _ = self.self_attn(src, src, src, )

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class intraAFL_Block(nn.Module):

    def __init__(self, input_dim, nhead, c, dropout, dim_feedforward=2048):
        super(intraAFL_Block, self).__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, nhead, dropout=dropout, batch_first=True, bias=True)
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(input_dim, dim_feedforward, )
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.expand = nn.Conv2d(1, c, kernel_size=1)
        self.pooling = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        self.proj = nn.Linear(c, input_dim)

        self.activation = F.relu

    def forward(self, src):
        src2, attnScore = self.self_attn(src, src, src, )
        attnScore = attnScore[:, np.newaxis]

        edge_emb = self.expand(attnScore)
        # edge_emb = self.pooling(edge_emb)
        w = edge_emb
        w = w.softmax(dim=-1)
        w = (w * edge_emb).sum(-1).transpose(-1, -2)
        w = self.proj(w)
        src2 = src2 + w

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class intraAFL(nn.Module):
    def __init__(self, input_dim, c, config):
        super(intraAFL, self).__init__()
        self.input_dim = input_dim
        self.num_block = config.get('NO_IntraAFL', 3)
        NO_head = config.get('NO_head', 1)
        dropout = config.get('dropout', 0.1)

        self.blocks = nn.ModuleList(
            [intraAFL_Block(input_dim=input_dim, nhead=NO_head, c=c, dropout=dropout) for _ in range(self.num_block)])

        self.fc = DeepFc(input_dim, input_dim)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        out = out.squeeze()
        out = self.fc(out)
        return out


class RegionFusion(nn.Module):
    def __init__(self, input_dim, config):
        super(RegionFusion, self).__init__()
        self.input_dim = input_dim
        self.num_block = config.get('NO_RegionFusion', 3)
        NO_head = config.get('NO_head', 1)
        dropout =config.get('dropout', 0.1)

        self.blocks = nn.ModuleList(
            [RegionFusionBlock(input_dim=input_dim, nhead=NO_head, dropout=dropout) for _ in range(self.num_block)])

        self.fc = DeepFc(input_dim, input_dim)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        out = out.squeeze()
        out = self.fc(out)
        return out


class interAFL_Block(nn.Module):

    def __init__(self, d_model, S):
        super(interAFL_Block, self).__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, queries):
        attn = self.mk(queries)
        attn = self.softmax(attn)
        attn = attn / torch.sum(attn, dim=2, keepdim=True)
        out = self.mv(attn)

        return out


class interAFL(nn.Module):
    def __init__(self, input_dim, d_m, config):
        super(interAFL, self).__init__()
        self.input_dim = input_dim
        self.num_block = config.get('NO_InterAFL', 3)

        self.blocks = nn.ModuleList(
            [interAFL_Block(input_dim, d_m) for _ in range(self.num_block)])

        self.fc = DeepFc(input_dim, input_dim)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        out = out.squeeze()
        out = self.fc(out)
        return out


class ViewFusion(nn.Module):
    def __init__(self, emb_dim, out_dim):
        super(ViewFusion, self).__init__()
        self.W = nn.Conv1d(emb_dim, out_dim, kernel_size=1, bias=False)
        self.f1 = nn.Conv1d(out_dim, 1, kernel_size=1)
        self.f2 = nn.Conv1d(out_dim, 1, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, src):
        seq_fts = self.W(src)
        f_1 = self.f1(seq_fts)
        f_2 = self.f2(seq_fts)
        logits = f_1 + f_2.transpose(1, 2)
        coefs = torch.mean(self.act(logits), dim=-1)
        coefs = torch.mean(coefs, dim=0)
        coefs = F.softmax(coefs, dim=-1)
        return coefs


class HAFusion_Model(nn.Module):
    def __init__(self, poi_dim, landUse_dim, input_dim, output_dim, d_prime, d_m, c, config):
        super(HAFusion_Model, self).__init__()
        self.input_dim = input_dim
        self.densePOI2 = nn.Linear(poi_dim, input_dim)
        self.denseLandUse3 = nn.Linear(landUse_dim, input_dim)

        self.encoderPOI = intraAFL(input_dim, c, config)
        self.encoderLandUse = intraAFL(input_dim, c, config)
        self.encoderMob = intraAFL(input_dim, c, config)

        self.regionFusionLayer = RegionFusion(input_dim, config)

        self.interViewEncoder = interAFL(input_dim, d_m, config)

        self.fc = DeepFc(input_dim, output_dim)

        self.para1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True) 
        self.para1.data.fill_(0.1)
        self.para2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True) 
        self.para2.data.fill_(0.9)

        self.viewFusionLayer = ViewFusion(input_dim, d_prime)

        self.activation = F.relu
        self.dropout = nn.Dropout(0.1)
        self.decoder_s = nn.Linear(output_dim, output_dim)  #
        self.decoder_t = nn.Linear(output_dim, output_dim)
        self.decoder_p = nn.Linear(output_dim, output_dim)  #
        self.decoder_l = nn.Linear(output_dim, output_dim)
        self.feature = None

    def forward(self, x):
        poi_emb, landUse_emb, mob_emb = x

        poi_emb = self.dropout(self.activation(self.densePOI2(poi_emb)))
        landUse_emb = self.dropout(self.activation(self.denseLandUse3(landUse_emb)))

        poi_emb = self.encoderPOI(poi_emb)
        landUse_emb = self.encoderLandUse(landUse_emb)
        mob_emb = self.encoderMob(mob_emb)

        out = torch.stack([poi_emb, landUse_emb, mob_emb])

        intra_view_embs = out
        out = out.transpose(0, 1)
        out = self.interViewEncoder(out)
        out = out.transpose(0, 1)
        p1 = self.para1 / (self.para1 + self.para2)
        p2 = self.para2 / (self.para1 + self.para2)
        out = out * p2 + intra_view_embs * p1
        # ---------------------------------------------

        out1 = out.transpose(0, 2)
        coef = self.viewFusionLayer(out1)
        temp_out = coef[0] * out[0] + coef[1] * out[1] + coef[2] * out[2]
        # --------------------------------------------------

        temp_out = temp_out[np.newaxis]
        temp_out = self.regionFusionLayer(temp_out)
        out = self.fc(temp_out)

        self.feature = out

        out_s = self.decoder_s(out)  # source embedding of regions
        out_t = self.decoder_t(out)  # destination embedding of regions
        out_p = self.decoder_p(out)  # poi embedding of regions
        out_l = self.decoder_l(out)  # landuse embedding of regions
        return out_s, out_t, out_p, out_l


    def out_feature(self):
        return self.feature