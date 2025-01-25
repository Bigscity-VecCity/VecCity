from logging import getLogger

import torch
import torch.nn as nn
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GATConv
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

from veccity.upstream.abstract_replearning_model import AbstractReprLearningModel


def random_mask(
    gps_assign_mat,
    route_assign_mat,
    gps_length,
    mask_token,
    mask_length=1,
    mask_prob=0.2,
):

    # mask route
    col_num = int(route_assign_mat.shape[1] / mask_length) + 1
    batch_size = route_assign_mat.shape[0]

    # mask的位置和padding的位置有重合，但整体mask概率无影响
    route_mask_pos = (
        torch.empty(
            (batch_size, col_num), dtype=torch.float32, device=route_assign_mat.device
        ).uniform_(0, 1)
        < mask_prob
    )

    route_mask_pos = torch.stack(
        sum([[col] * mask_length for col in route_mask_pos.t()], []), dim=1
    )

    # 截断
    if route_mask_pos.shape[1] > route_assign_mat.shape[1]:
        route_mask_pos = route_mask_pos[:, : route_assign_mat.shape[1]]

    masked_route_assign_mat = route_assign_mat.clone()
    masked_route_assign_mat[route_mask_pos] = mask_token

    # mask gps
    masked_gps_assign_mat = gps_assign_mat.clone()
    gps_mask_pos = []
    for idx, row in enumerate(gps_assign_mat):
        route_mask = route_mask_pos[idx]
        length_list = gps_length[idx]
        unpad_mask_pos_list = sum(
            [[mask] * length_list[_idx].item() for _idx, mask in enumerate(route_mask)],
            [],
        )
        pad_mask_pos_list = unpad_mask_pos_list + [
            torch.tensor(False).to(route_mask.device)
        ] * (gps_assign_mat.shape[1] - len(unpad_mask_pos_list))
        pad_mask_pos = torch.stack(pad_mask_pos_list)
        gps_mask_pos.append(pad_mask_pos)
    gps_mask_pos = torch.stack(gps_mask_pos, dim=0)
    masked_gps_assign_mat[gps_mask_pos] = mask_token
    # 获得每个gps点对应路段的长度

    return masked_route_assign_mat, masked_gps_assign_mat


def get_traj_match_loss(gps_traj_rep, route_traj_rep, model, batch_size=64, tau=0.07):
    gps_traj_rep = F.normalize(gps_traj_rep, dim=1)
    route_traj_rep = F.normalize(route_traj_rep, dim=1)

    # gps_traj_all = torch.cat([gps_traj_rep.t(), model.gps_queue.clone().detach()], dim=1) # 256 x 2048+64
    # route_traj_all = torch.cat([route_traj_rep.t(), model.route_queue.clone().detach()], dim=1)

    sim_g2r = gps_traj_rep @ route_traj_rep.t() / tau
    sim_r2g = route_traj_rep @ gps_traj_rep.t() / tau

    weight_g2r = F.softmax(sim_g2r, dim=1)
    weight_r2g = F.softmax(sim_r2g, dim=1)

    sim_g2r.fill_diagonal_(0)
    sim_r2g.fill_diagonal_(0)

    # select a negative route for each gps
    route_traj_rep_neg = []
    for i in range(batch_size):
        neg_idx = torch.multinomial(weight_g2r[i], 1).item()
        route_traj_rep_neg.append(route_traj_rep[neg_idx])
    route_traj_rep_neg = torch.stack(route_traj_rep_neg, dim=0)

    # select a negative gps for each route
    gps_traj_rep_neg = []
    for i in range(batch_size):
        neg_idx = torch.multinomial(weight_r2g[i], 1).item()
        gps_traj_rep_neg.append(gps_traj_rep[neg_idx])
    gps_traj_rep_neg = torch.stack(gps_traj_rep_neg, dim=0)

    # 每一个GR pair都有两个负样本 GR’ 和 G‘R，flat之后分为3组，GR,GR',G'R 64*3 x 256

    pos_pair = torch.cat([route_traj_rep, gps_traj_rep], dim=1)
    neg_pair1 = torch.cat([route_traj_rep_neg, gps_traj_rep], dim=1)
    neg_pair2 = torch.cat([route_traj_rep, gps_traj_rep_neg], dim=1)

    all_pair = torch.cat([pos_pair, neg_pair1, neg_pair2], dim=0)

    pred = model.matching_predictor(all_pair)  # 2 x 64*3

    label = torch.cat(
        [
            torch.ones(batch_size, dtype=torch.long),
            torch.zeros(2 * batch_size, dtype=torch.long),
        ],
        dim=0,
    ).cuda()  # 1 x 64*3
    loss = F.cross_entropy(pred, label)
    return loss


class JGRM(AbstractReprLearningModel):
    def __init__(self, config, data_feature):
        super(JGRM, self).__init__(config, data_feature)

        self.edge_index = data_feature.get("edge_index")
        self._logger = getLogger()

        self.device = config.get("device")
        self.vocab_size = config.get("vocab_size", 100)  # 路段数量
        self.gps_feat_num = config.get("gps_feat_num", 2)  # 2

        self.gps_embed_size = config.get("gps_embed_size", 100)
        self.road_embed_size = config.get("road_embed_size", 100)
        self.route_embed_size = config.get("route_embed_size", 100)

        self.drop_edge_rate = config.get("drop_edge_rate", 0.0)
        self.drop_route_rate = config.get("drop_route_rate", 0.0)
        self.drop_road_rate = config.get("drop_road_rate", 0.0)

        self.hidden_size = config.get("hidden_size", 6)

        self.route_max_len = config.get("route_max_len", 100)

        self.mask_length = config.get("mask_length", 2)
        self.mask_prob = config.get("mask_prob", 0.2)

        self.mode = config.get("mode", "p")
        self.batch_size = config.get("batch_size", 32)

        # node embedding
        self.route_padding_vec = torch.zeros(
            1, self.road_embed_size, requires_grad=True
        ).to(self.device)
        self.node_embedding = nn.Embedding(self.vocab_size, self.road_embed_size)
        self.node_embedding.weight = torch.nn.Parameter(
            data_feature.get("init_road_emb")
        )
        self.node_embedding.requires_grad_(True)

        # time embedding 考虑加法, 保证 embedding size一致
        self.minute_embedding = nn.Embedding(
            1440 + 1, self.route_embed_size
        )  # 0 是mask位
        self.week_embedding = nn.Embedding(7 + 1, self.route_embed_size)  # 0 是mask位
        self.delta_embedding = IntervalEmbedding(
            100, self.route_embed_size
        )  # -1 是mask位

        # route encoding
        self.graph_encoder = GraphEncoder(self.road_embed_size, self.route_embed_size)
        self.position_embedding1 = nn.Embedding(
            self.route_max_len, self.route_embed_size
        )
        self.fc1 = nn.Linear(
            self.route_embed_size, self.hidden_size
        )  # route fuse time ffn
        self.route_encoder = TransformerModel(
            self.hidden_size, 8, self.hidden_size, 4, self.drop_route_rate
        )

        # gps encoding
        self.gps_linear = nn.Linear(self.gps_feat_num, self.gps_embed_size)
        self.gps_intra_encoder = nn.GRU(
            self.gps_embed_size,
            self.gps_embed_size,
            bidirectional=True,
            batch_first=True,
        )  # 路段内建模
        self.gps_inter_encoder = nn.GRU(
            self.gps_embed_size,
            self.gps_embed_size,
            bidirectional=True,
            batch_first=True,
        )  # 路段间建模

        # cl project head
        self.gps_proj_head = nn.Linear(2 * self.gps_embed_size, self.hidden_size)
        self.route_proj_head = nn.Linear(self.hidden_size, self.hidden_size)

        # shared transformer
        self.position_embedding2 = nn.Embedding(self.route_max_len, self.hidden_size)
        self.modal_embedding = nn.Embedding(2, self.hidden_size)
        self.fc2 = nn.Linear(
            self.hidden_size, self.hidden_size
        )  # shared transformer position transform
        self.sharedtransformer = TransformerModel(
            self.hidden_size, 4, self.hidden_size, 2, self.drop_road_rate
        )

        # mlm classifier head
        self.gps_mlm_head = nn.Linear(self.hidden_size, self.vocab_size)
        self.route_mlm_head = nn.Linear(self.hidden_size, self.vocab_size)

        # matching
        self.matching_predictor = nn.Linear(self.hidden_size * 2, 2)
        self.register_buffer("gps_queue", torch.randn(self.hidden_size, 2048))
        self.register_buffer("route_queue", torch.randn(self.hidden_size, 2048))

        self.image_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.route_queue, dim=0)

    def encode_graph(self, drop_rate=0.0):
        node_emb = self.node_embedding.weight
        edge_index = dropout_edge(self.edge_index, p=drop_rate)[0]
        node_enc = self.graph_encoder(node_emb, edge_index)
        return node_enc

    def encode_route(self, route_data, route_assign_mat, masked_route_assign_mat):
        # 返回路段表示和轨迹的表示
        if self.mode == "p":
            lookup_table = torch.cat(
                [self.node_embedding.weight, self.route_padding_vec], 0
            )
        else:
            node_enc = self.encode_graph(self.drop_edge_rate)
            lookup_table = torch.cat([node_enc, self.route_padding_vec], 0)

        # 先对原始序列进行mask，然后再进行序列建模，防止信息泄露
        batch_size, max_seq_len = masked_route_assign_mat.size()

        src_key_padding_mask = route_assign_mat == self.vocab_size
        pool_mask = (1 - src_key_padding_mask.int()).unsqueeze(-1)  # 0 为padding位

        route_emb = torch.index_select(
            lookup_table, 0, masked_route_assign_mat.int().view(-1)
        ).view(batch_size, max_seq_len, -1)

        # time embedding
        if route_data is None:  # node evaluation的时候使用
            # 取时间表示的平均作为无时间特征输入时的表示
            week_emb = self.week_embedding.weight.detach()[1:].mean(dim=0)
            min_emb = self.minute_embedding.weight.detach()[1:].mean(dim=0)
            delta_emb = self.minute_embedding.weight.detach()[1:].mean(dim=0)
        else:
            week_data = route_data[:, :, 0].long()
            min_data = route_data[:, :, 1].long()
            delta_data = route_data[:, :, 2].float()
            week_emb = self.week_embedding(week_data)
            min_emb = self.minute_embedding(min_data)
            delta_emb = self.delta_embedding(delta_data)

        # position embedding
        position = torch.arange(route_emb.shape[1]).long().to(self.device)
        pos_emb = position.unsqueeze(0).repeat(route_emb.shape[0], 1)  # (S,) -> (B, S)
        pos_emb = self.position_embedding1(pos_emb)

        # fuse info
        route_emb = route_emb + pos_emb + week_emb + min_emb + delta_emb
        route_emb = self.fc1(route_emb)
        route_enc = self.route_encoder(
            route_emb, None, src_key_padding_mask
        )  # mask 被在这里处理，mask不参与计算attention
        route_enc = torch.where(
            torch.isnan(route_enc), torch.full_like(route_enc, 0), route_enc
        )  # 将nan变为0,防止溢出

        route_unpooled = route_enc * pool_mask.repeat(
            1, 1, route_enc.shape[-1]
        )  # (batch_size,max_len,feat_num)

        # 对于单路段mask，可能存在整个route都被mask掉的情况，此时pool_mask.sum(1)中有0值，令其最小值为1防止0除
        route_pooled = route_unpooled.sum(1) / pool_mask.sum(1).clamp(
            min=1
        )  # (batch_size, feat_num)

        return route_unpooled, route_pooled

    def encode_gps(
        self, gps_data, masked_gps_assign_mat, masked_route_assign_mat, gps_length
    ):
        # gps_data 先输入 gps_encoder, 输出每个step的output，选择路段对应位置的gps点的output进行pooling作为路段的表示
        gps_data = self.gps_linear(gps_data)

        # mask features
        gps_src_key_padding_mask = masked_gps_assign_mat == self.vocab_size
        gps_mask_mat = (
            (1 - gps_src_key_padding_mask.int())
            .unsqueeze(-1)
            .repeat(1, 1, gps_data.shape[-1])
        )  # 0 为padding位
        masked_gps_data = gps_data * gps_mask_mat  # (batch_size,gps_max_len,feat_num)

        # flatten gps data 便于进行路段内gru的并行
        flattened_gps_data, route_length = self.gps_flatten(
            masked_gps_data, gps_length
        )  # flattened_gps_data (road_num, max_pt_len ,gps_fea_size)
        _, gps_emb = self.gps_intra_encoder(
            flattened_gps_data
        )  # gps_emb (1, road_num, gps_embed_size) # 不输入hidden默认输入全0为序列的hidden state
        gps_emb = gps_emb[-1]  # 只保留前向的表示
        # gps_emb = torch.cat([gps_emb[0].squeeze(0), gps_emb[1].squeeze(0)],dim=-1) # 前后向表示拼接

        # stack gps emb 便于进行路段间gru的计算
        stacked_gps_emb = self.route_stack(
            gps_emb, route_length
        )  # stacked_gps_emb (batch_size, max_route_len, gps_embed_size)
        gps_emb, _ = self.gps_inter_encoder(
            stacked_gps_emb
        )  # (batch_size, max_route_len, 2*gps_embed_size) # 不输入hidden默认输入全0为序列的hidden state

        route_src_key_padding_mask = (
            masked_route_assign_mat == self.vocab_size
        ).transpose(0, 1)
        route_pool_mask = (
            (1 - route_src_key_padding_mask.int()).transpose(0, 1).unsqueeze(-1)
        )  # 包含mask的长度
        # 对于单路段mask，可能存在整个route都被mask掉的情况，此时pool_mask.sum(1)中有0值，令其最小值为1防止0除
        gps_pooled = gps_emb.sum(1) / route_pool_mask.sum(1).clamp(
            min=1
        )  # mask 后的有值的路段数量，比路段长度要短
        gps_unpooled = gps_emb

        return gps_unpooled, gps_pooled

    def route_stack(self, gps_emb, route_length):
        # flatten_gps_data tensor = (real_len, max_gps_in_route_len, emb_size)
        # route_length dict = { key:tid, value: road_len }
        values = list(route_length.values())
        route_max_len = max(values)
        data_list = []
        for idx in range(len(route_length)):
            start_idx = sum(values[:idx])
            end_idx = sum(values[: idx + 1])
            data = gps_emb[start_idx:end_idx]
            data_list.append(data)

        stacked_gps_emb = rnn_utils.pad_sequence(
            data_list, padding_value=0, batch_first=True
        )

        return stacked_gps_emb

    def gps_flatten(self, gps_data, gps_length):
        # 把gps_data按照gps_assign_mat做形变，把每个路段上的gps点单独拿出来，拼成一个新的tensor (road_num, gps_max_len, gps_feat_num)，
        # 该tensor用于输入GRU进行并行计算
        traj_num, gps_max_len, gps_feat_num = gps_data.shape
        flattened_gps_list = []
        route_index = {}
        for idx in range(traj_num):
            gps_feat = gps_data[idx]  # (max_len, feat_num)
            length_list = gps_length[
                idx
            ]  # (max_len, 1) [7,9,12,1,0,0,0,0,0,0] # padding_value = 0
            # 遍历每个轨迹中的路段
            for _idx, length in enumerate(length_list):
                if length != 0:
                    start_idx = sum(length_list[:_idx])
                    end_idx = start_idx + length_list[_idx]
                    route_index[idx] = route_index.get(idx, 0) + 1
                    road_feat = gps_feat[start_idx:end_idx]
                    flattened_gps_list.append(road_feat)

        flattened_gps_data = rnn_utils.pad_sequence(
            flattened_gps_list, padding_value=0, batch_first=True
        )  # (road_num, gps_max_len, gps_feat_num)

        return flattened_gps_data, route_index

    def encode_joint(
        self,
        route_road_rep,
        route_traj_rep,
        gps_road_rep,
        gps_traj_rep,
        route_assign_mat,
    ):
        max_len = torch.max((route_assign_mat != self.vocab_size).int().sum(1)).item()
        max_len = max_len * 2 + 2
        data_list = []
        mask_list = []
        route_length = [
            length[length != self.vocab_size].shape[0] for length in route_assign_mat
        ]

        modal_emb0 = self.modal_embedding(torch.tensor(0).to(self.device))
        modal_emb1 = self.modal_embedding(torch.tensor(1).to(self.device))

        for i, length in enumerate(route_length):
            route_road_token = route_road_rep[i][:length]
            gps_road_token = gps_road_rep[i][:length]
            route_cls_token = route_traj_rep[i].unsqueeze(0)
            gps_cls_token = gps_traj_rep[i].unsqueeze(0)

            # position
            position = torch.arange(length + 1).long().to(self.device)
            pos_emb = self.position_embedding2(position)

            # update route_emb
            route_emb = torch.cat([route_cls_token, route_road_token], dim=0)
            modal_emb = modal_emb0.unsqueeze(0).repeat(length + 1, 1)
            route_emb = route_emb + pos_emb + modal_emb
            route_emb = self.fc2(route_emb)

            # update gps_emb
            gps_emb = torch.cat([gps_cls_token, gps_road_token], dim=0)
            modal_emb = modal_emb1.unsqueeze(0).repeat(length + 1, 1)
            gps_emb = gps_emb + pos_emb + modal_emb
            gps_emb = self.fc2(gps_emb)

            data = torch.cat([gps_emb, route_emb], dim=0)
            data_list.append(data)

            mask = torch.tensor([False] * data.shape[0]).to(
                self.device
            )  # mask的位置为true
            mask_list.append(mask)

        joint_data = rnn_utils.pad_sequence(
            data_list, padding_value=0, batch_first=True
        )
        mask_mat = rnn_utils.pad_sequence(
            mask_list, padding_value=True, batch_first=True
        )

        joint_emb = self.sharedtransformer(joint_data, None, mask_mat)

        # 每一行的0 和 length+1 对应的是 gps_traj_rep 和 route_traj_rep
        gps_traj_rep = joint_emb[:, 0]
        route_traj_rep = torch.stack(
            [joint_emb[i, length + 1] for i, length in enumerate(route_length)], dim=0
        )

        gps_road_rep = rnn_utils.pad_sequence(
            [joint_emb[i, 1 : length + 1] for i, length in enumerate(route_length)],
            padding_value=0,
            batch_first=True,
        )
        route_road_rep = rnn_utils.pad_sequence(
            [
                joint_emb[i, length + 2 : 2 * length + 2]
                for i, length in enumerate(route_length)
            ],
            padding_value=0,
            batch_first=True,
        )

        return gps_road_rep, gps_traj_rep, route_road_rep, route_traj_rep

    def forward(
        self,
        route_data,
        masked_route_assign_mat,
        gps_data,
        masked_gps_assign_mat,
        route_assign_mat,
        gps_length,
    ):
        gps_road_rep, gps_traj_rep = self.encode_gps(
            gps_data, masked_gps_assign_mat, masked_route_assign_mat, gps_length
        )
        route_road_rep, route_traj_rep = self.encode_route(
            route_data, route_assign_mat, masked_route_assign_mat
        )
        (
            gps_road_joint_rep,
            gps_traj_joint_rep,
            route_road_joint_rep,
            route_traj_joint_rep,
        ) = self.encode_joint(
            route_road_rep, route_traj_rep, gps_road_rep, gps_traj_rep, route_assign_mat
        )
        return (
            gps_road_rep,
            gps_traj_rep,
            route_road_rep,
            route_traj_rep,
            gps_road_joint_rep,
            gps_traj_joint_rep,
            route_road_joint_rep,
            route_traj_joint_rep,
        )

    def predict(self, batch):
        gps_data, gps_assign_mat, route_data, route_assign_mat, gps_length = batch
        masked_route_assign_mat, masked_gps_assign_mat = random_mask(
            gps_assign_mat,
            route_assign_mat,
            gps_length,
            self.vocab_size,
            self.mask_length,
            self.mask_prob,
        )

        (
            route_data,
            masked_route_assign_mat,
            gps_data,
            masked_gps_assign_mat,
            route_assign_mat,
            gps_length,
        ) = (
            route_data.to(self.device),
            masked_route_assign_mat.to(self.device),
            gps_data.to(self.device),
            masked_gps_assign_mat.to(self.device),
            route_assign_mat.to(self.device),
            gps_length.to(self.device),
        )

        (
            gps_road_rep,
            gps_traj_rep,
            route_road_rep,
            route_traj_rep,
            gps_road_joint_rep,
            gps_traj_joint_rep,
            route_road_joint_rep,
            route_traj_joint_rep,
        ) = self.forward(
            route_data,
            masked_route_assign_mat,
            gps_data,
            masked_gps_assign_mat,
            route_assign_mat,
            gps_length,
        )

        return (
            gps_road_rep,
            gps_traj_rep,
            route_road_rep,
            route_traj_rep,
            gps_road_joint_rep,
            gps_traj_joint_rep,
            route_road_joint_rep,
            route_traj_joint_rep,
            masked_route_assign_mat,
            masked_gps_assign_mat,
        )

    def calculate_loss(self, batch):
        """
        Calculate the loss for the model.

        Args:
            batch: A tuple of (gps_data, gps_assign_mat, route_data, route_assign_mat, gps_length)

        Return:
           loss: The loss for the model.
        """

        gps_data, gps_assign_mat, route_data, route_assign_mat, gps_length = batch
        (
            gps_road_rep,
            gps_traj_rep,
            route_road_rep,
            route_traj_rep,
            gps_road_joint_rep,
            gps_traj_joint_rep,
            route_road_joint_rep,
            route_traj_joint_rep,
            masked_route_assign_mat,
            masked_gps_assign_mat,
        ) = self.predict(batch)

        # flatten road_rep
        mat2flatten = {}
        y_label = []
        route_length = (route_assign_mat != self.vocab_size).int().sum(1)
        (
            gps_road_list,
            route_road_list,
            gps_road_joint_list,
            route_road_joint_list,
        ) = ([], [], [], [])
        now_flatten_idx = 0
        for i, length in enumerate(route_length):
            y_label.append(
                route_assign_mat[i, :length]
            )  # route 和 gps mask 位置是一样的
            gps_road_list.append(gps_road_rep[i, :length])
            route_road_list.append(route_road_rep[i, :length])
            gps_road_joint_list.append(gps_road_joint_rep[i, :length])
            route_road_joint_list.append(route_road_joint_rep[i, :length])
            for l in range(length):
                mat2flatten[(i, l)] = now_flatten_idx
                now_flatten_idx += 1

        y_label = torch.cat(y_label, dim=0)
        gps_road_rep = torch.cat(gps_road_list, dim=0)
        route_road_rep = torch.cat(route_road_list, dim=0)
        gps_road_joint_rep = torch.cat(gps_road_joint_list, dim=0)
        route_road_joint_rep = torch.cat(route_road_joint_list, dim=0)

        # project rep into the same space
        gps_traj_rep = self.gps_proj_head(gps_traj_rep)
        route_traj_rep = self.route_proj_head(route_traj_rep)

        # (GRM LOSS) get gps & route rep matching loss
        tau = 0.07
        match_loss = get_traj_match_loss(
            gps_traj_rep, route_traj_rep, self, self.batch_size, tau
        )

        # prepare label and mask_pos
        masked_pos = torch.nonzero(route_assign_mat != masked_route_assign_mat)
        masked_pos = [mat2flatten[tuple(pos.tolist())] for pos in masked_pos]
        y_label = y_label[masked_pos].long()

        # (MLM 1 LOSS) get gps rep road loss
        gps_mlm_pred = self.gps_mlm_head(gps_road_joint_rep)  # project head 也会被更新
        masked_gps_mlm_pred = gps_mlm_pred[masked_pos]
        gps_mlm_loss = nn.CrossEntropyLoss()(masked_gps_mlm_pred, y_label)

        # (MLM 2 LOSS) get route rep road loss
        route_mlm_pred = self.route_mlm_head(
            route_road_joint_rep
        )  # project head 也会被更新
        masked_route_mlm_pred = route_mlm_pred[masked_pos]
        route_mlm_loss = nn.CrossEntropyLoss()(masked_route_mlm_pred, y_label)

        # MLM 1 LOSS + MLM 2 LOSS + GRM LOSS
        loss = (route_mlm_loss + gps_mlm_loss + 2 * match_loss) / 3

        return loss

    def encode(self, batch):
        (
            gps_road_rep,
            gps_traj_rep,
            route_road_rep,
            route_traj_rep,
            gps_road_joint_rep,
            gps_traj_joint_rep,
            route_road_joint_rep,
            route_traj_joint_rep,
            masked_route_assign_mat,
            masked_gps_assign_mat,
        ) = self.predict(batch)
        return route_traj_joint_rep

    def encode_all(self, batch):
        (
            gps_road_rep,
            gps_traj_rep,
            route_road_rep,
            route_traj_rep,
            gps_road_joint_rep,
            gps_traj_joint_rep,
            route_road_joint_rep,
            route_traj_joint_rep,
            masked_route_assign_mat,
            masked_gps_assign_mat,
        ) = self.predict(batch)
        return (
            gps_road_rep,
            gps_traj_rep,
            route_road_rep,
            route_traj_rep,
            gps_road_joint_rep,
            gps_traj_joint_rep,
            route_road_joint_rep,
            route_traj_joint_rep,
        )


# GAT
class GraphEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphEncoder, self).__init__()
        # update road edge features using GAT
        self.layer1 = GATConv(input_size, output_size)
        self.layer2 = GATConv(input_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.activation(self.layer1(x, edge_index))
        x = self.activation(self.layer2(x, edge_index))
        return x


class TransformerModel(nn.Module):  # vanilla transformer
    def __init__(self, input_size, num_heads, hidden_size, num_layers, dropout=0.3):
        super(TransformerModel, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(
            input_size, num_heads, hidden_size, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src, src_mask, src_key_padding_mask):
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        return output


# Continuous time embedding
class IntervalEmbedding(nn.Module):
    def __init__(self, num_bins, hidden_size):
        super(IntervalEmbedding, self).__init__()
        self.layer1 = nn.Linear(1, num_bins)
        self.emb = nn.Embedding(num_bins, hidden_size)
        self.activation = nn.Softmax()

    def forward(self, x):
        logit = self.activation(self.layer1(x.unsqueeze(-1)))
        output = logit @ self.emb.weight
        return output
