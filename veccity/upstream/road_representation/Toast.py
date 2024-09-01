import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from veccity.upstream.road_representation.utils import *
from veccity.upstream.abstract_replearning_model import AbstractReprLearningModel
import os

from gensim.models import Word2Vec
import gensim
import numpy as np
from tqdm import tqdm
from logging import getLogger
from torch.utils.data import DataLoader
from veccity.utils import ensure_dir


class Toast(AbstractReprLearningModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self.iter = config.get('max_epoch', 5)
        self.w2v_model = Word2Vec_SG(config,data_feature)
        self.model = BertModel4Pretrain(config,data_feature)
        self.device = config.get('device')
        self.load_init = config.get('load_init')
        self.data = data_feature.get('dataloader')
        self.batch_size = config.get('batch_size',64)
        self.n_workers = config.get('n_workers',1)
        self.dataloader= DataLoader(self.data, batch_size=self.batch_size, num_workers=self.n_workers)
        self.model_name = config.get('model', '')
        self.exp_id = config.get('exp_id', None)
        self.dataset = config.get('dataset', '')
        self.output_dim=config.get('embedding_size',128)
        self.lr=config.get('lr',0.005)
        
        ensure_dir( './veccity/cache/model_cache/{}'.format(self.model_name))
        self.vocab_embed_path = './veccity/cache/model_cache/{}/w2v_{}.pt'.format(self.model_name,self.dataset)
        self.model_cache_file = './veccity/cache/{}/model_cache/embedding_{}_{}_{}.m'. \
            format(self.exp_id, self.model_name, self.dataset, self.output_dim)
        self.road_embedding_path = './veccity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model_name, self.dataset, self.output_dim)
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.criterion1 = nn.CrossEntropyLoss(reduction='none')
        self.criterion2 = nn.CrossEntropyLoss()
        
        # self.model.to(self.device)
        self.get_w2v_embed()
        self.model.init_token_embed(self.w2v_model.get_list_vector())
        self.model.to(self.device)


    def calculate_loss(self,data,seen_batch):
        mask_lm_output, next_sent_output  = self.model.forward(data["seq"].to(self.device), data["padding_masks"].to(self.device), data['target_masks'].to(self.device), data['length'].to(self.device))
        next_loss = self.criterion2(next_sent_output, data["is_traj"].long().to(self.device))
        targets=data["targets"].squeeze(-1)[data['target_masks'].squeeze()]
        mask_lm_output = mask_lm_output[data['target_masks'].squeeze()] 
        mask_loss = self.criterion1(mask_lm_output, targets.to(self.device))

        mask_loss = mask_loss.mean()
        loss = next_loss + mask_loss
        return loss
    
    def encode_sequence(self, batch):
        seq=batch['seq'][...,0].to(self.device)
        padding_masks=batch['padding_masks'].to(self.device)
        return self.model.encode_sequence(seq,padding_masks)     

    def get_w2v_embed(self):
        if self.load_init:
            if not os.path.exists(self.vocab_embed_path):
                for i in range(4):
                    self.w2v_model.train(i, mode='pretrain')
                    
                    self.w2v_model.save_model(i, self.vocab_embed_path)
            else:
                print("Load from pretrained traffic context aware skip-gram model")
                self.w2v_model.load_model(self.vocab_embed_path)
        else:
            for i in range(4):
                self.w2v_model.train(i, mode='pretrain')
                
                self.w2v_model.save_model(i, self.vocab_embed_path)

    def get_static_embedding(self):
        node_embedding=self.model.transformer.embed.tok_embed.weight.data.cpu().detach().numpy()
        return node_embedding
    
    def run(self):
        """
        Args:
            data : input of tradition model

        Returns:
            output of tradition model
        """
        if not self.config.get('train') and os.path.exists(self.road_embedding_path):
            return
        if self.load_init:
            if not os.path.exists(self.vocab_embed_path):
                for i in range(4):
                    self.w2v_model.train(i, mode='pretrain')
                    
                    self.w2v_model.save_model(i, self.vocab_embed_path)
            else:
                print("Load from pretrained traffic context aware skip-gram model")
                self.w2v_model.load_model(self.vocab_embed_path)
        else:
            for i in range(4):
                self.w2v_model.train(i, mode='pretrain')
                
                self.w2v_model.save_model(i, self.vocab_embed_path)
        self.model.init_token_embed(self.w2v_model.get_list_vector())
        self.model.to(self.device)
        self.dataloader.dataset.gen_new_walks(num_walks=1000)
        self.train_step=0
        for epoch in tqdm(range(self.iter)):
            self.model.train()
            str_code = "train"

            avg_loss = 0.0
            total_correct = 0
            total_element = 0
            for i, data in enumerate(self.dataloader):
                data = {key: value.to(self.device) for key, value in data.items()}
                # print(data)
                mask_lm_output, next_sent_output  = self.model.forward(data["traj_input"].to(self.device), data["input_mask"].to(self.device), data['masked_pos'].to(self.device), data['length'].to(self.device))

                next_loss = self.criterion2(next_sent_output, data["is_traj"].long())
                mask_loss = self.criterion1(mask_lm_output.transpose(1, 2), data["masked_tokens"])

                mask_loss = (mask_loss * data['masked_weights'].float()).mean()

                loss = next_loss + mask_loss

            
                self.train_step += 1
                self.optim.zero_grad()

                loss.backward()
                self.optim.step()


                correct = next_sent_output.argmax(dim=-1).eq(data["is_traj"].long()).sum().item()
                avg_loss += loss.item()
                total_correct += correct
                total_element += data["is_traj"].nelement()

                if i % 100 == 0:
                    print("Epoch: {}, iter {} loss: {}, masked traj loss {:.3f}, judge traj loss {:.3f}".format(epoch, i, loss.item(), mask_loss.item(), next_loss.item()))
                    
            if (epoch + 1) % 20 == 0:
                self.save_model()
            print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(self.dataloader), "total_acc=",
                total_correct * 100.0 / total_element)
        
        self.save_model()



    def save_model(self):
        """
        Args:
            cache_name : path to save parameters
        """
        # save road embedding
        node_embedding=self.model.transformer.embed.tok_embed.weight.cpu().detach().numpy()
        node_embedding=node_embedding[2:]
        np.save(self.road_embedding_path,node_embedding)
        print("EP: Embedding Saved on:{}".format(self.road_embedding_path))

        # save transformer encoder
        
        torch.save(self.model.state_dict(),
                   self.model_cache_file)

        print("EP: Model Saved on:{}".format(self.model_cache_file))

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    def __init__(self, dim, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    def __init__(self, vocab_size,dim,max_len,p_dropout):
        super(Embeddings, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, dim) # token embeddind
        self.pos_embed = nn.Embedding(max_len, dim) # position embedding
        self.norm = LayerNorm(dim)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x) # (S,) -> (B, S)
        e = self.tok_embed(x) + self.pos_embed(pos)
        
            
        return self.drop(self.norm(e))


def split_last(x, shape):
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, dim,n_heads,p_dropout):
        super(MultiHeadedSelfAttention, self).__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(p_dropout)
        self.scores = None # for visualization
        self.n_heads = n_heads

    def forward(self, x, mask):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])

        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))

        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim,dim_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, dim_ff)
        self.fc2 = nn.Linear(dim_ff, dim)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim,p_dropout,n_heads,dim_ff):
        super(Block, self).__init__()
        self.attn = MultiHeadedSelfAttention(dim,n_heads,p_dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = LayerNorm(dim)
        self.pwff = PositionWiseFeedForward(dim,dim_ff)
        self.norm2 = LayerNorm(dim)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


class Transformer(nn.Module):
    def __init__(self, hidden_dim,layers,vocab_size,dim,max_len,p_dropout,n_heads,dim_ff):
        super(Transformer, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.embed = Embeddings(vocab_size,dim,max_len,p_dropout)
        self.blocks = nn.ModuleList([Block(dim,p_dropout,n_heads,dim_ff) for _ in range(layers)])

    def forward(self, x, mask):
        
        h = self.fc(self.embed(x))
        for block in self.blocks:
            h = block(h, mask)
        return h


class SkipGramNeg(nn.Module):
    def __init__(self, vocab_size, emb_dim, type_num=13, type_dim=32):
        super(SkipGramNeg, self).__init__()
        self.input_emb = nn.Embedding(vocab_size, emb_dim)
        # self.output_emb = nn.Embedding(vocab_size, emb_dim)
        self.output_emb = nn.Embedding(vocab_size, emb_dim + type_num)
        self.type_pred = nn.Linear(emb_dim, type_num, bias=False)
        self.log_sigmoid = nn.LogSigmoid()

        initrange = (2.0 / (vocab_size + emb_dim)) ** 0.5  # Xavier init
        self.input_emb.weight.data.uniform_(-initrange, initrange)
        self.output_emb.weight.data.uniform_(-0, 0)

    def load_init_embed(self, w2v_file):
        w2v = Word2Vec.load(w2v_file)
        w2v_dict = {}
        for idx, key in enumerate(w2v.wv.vocab):
            w2v_dict[int(key)] = w2v.wv[key]

        sorted_w2v = {k: v for k, v in sorted(w2v_dict.items(), key=lambda item: item[0])}

        embed = []
        for k, v in sorted_w2v.items():
            embed.append(v)

        embed = torch.FloatTensor(embed)
        print("load embedding from initial model, shape {}".format(embed.shape))

        token_vocab = self.input_embe.weight.shape[0]
        if embed.shape[0] < token_vocab:
            self.input_embe.weight.data[token_vocab - embed.shape[0]:] = embed
            print(self.input_embe.weight.shape)
        else:
            self.input_embe.weight.data = embed


    def forward(self, target_input, type_input, context, types, neg, type_mask):

        v = self.input_emb(target_input)
        u = self.output_emb(context)

        type_pred = self.type_pred(v)

        type_loss = F.binary_cross_entropy_with_logits(type_pred, types, weight=type_mask)

        # positive_val: [batch_size]
        v_cat = torch.cat((v, torch.sigmoid(type_pred)), dim=1)
        positive_val = self.log_sigmoid(torch.sum(u * v_cat, dim=1)).squeeze()

        u_hat = self.output_emb(neg)
        neg_vals = torch.bmm(u_hat, v_cat.unsqueeze(2)).squeeze(2)
        neg_val = self.log_sigmoid(-torch.sum(neg_vals, dim=1)).squeeze()

        loss = positive_val + neg_val
        return -loss.mean(), type_loss

    def init_token_embed_from_bert(self, embed_bert):
        if embed_bert is not None:
            self.input_emb.weight.data = embed_bert

class Word2Vec_SG:
    def __init__(self, config, data_feature):

        self.walker = RandomWalker(config,data_feature)
        self.vocab = data_feature.get('vocab')
        self.word2index = data_feature.get('node2id') 
        self.index2word = data_feature.get('id2node')
        self.road_lengths = data_feature.get('road_lengths')
        self.type_num = data_feature.get('type_num',12)
        nodes = list(self.word2index.keys())
        self.vocabs = [self.word2index[node]+2 for node in nodes]
        self.embedding_size = config.get('embedding_size',64)
        self.device = config.get('device')
        self.lr = config.get('w2v_lr',0.1)
        self.num_walks = config.get('num_walks', 20)

        self.model = SkipGramNeg(self.vocab.vocab_size, self.embedding_size,self.type_num)
        self.optim = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        


    def train(self, epoch, mode, embed_bert=None):
        self.model.init_token_embed_from_bert(embed_bert)
        self.model.to(self.device)
        self.sentences = self.walker.generate_sentences_bert_type(num_walks=20)
        self.iteration(epoch, mode)


    def iteration(self, epoch, mode='pretrain', skip_window=5, num_neg=10):

        avg_loss = 0
        pipeline = DataPipeline(self.sentences, self.vocabs, self.index2word, self.road_lengths,self.type_num)
        step = 0

        if mode == 'pretrain':
            while pipeline.sen_index <= len(self.sentences):
                # batch_inputs, batch_labels = pipeline.generate_batch_one(skip_window)
                batch_inputs, input_types, batch_types, batch_labels, batch_type_masks = pipeline.generate_batch(skip_window)
                batch_neg = pipeline.get_neg_data(num_neg, batch_inputs)
                # print(batch_inputs, batch_types)
                # print(len(batch_inputs), len(batch_types))
                batch_inputs = torch.LongTensor(batch_inputs).to(self.device)
                input_types = torch.LongTensor(input_types).to(self.device)
                batch_labels = torch.LongTensor(batch_labels).to(self.device)
                batch_neg = torch.LongTensor(batch_neg).to(self.device)
                batch_types = torch.Tensor(batch_types).to(self.device)
                batch_type_masks = torch.Tensor(batch_type_masks).to(self.device)
                # print(batch_inputs.shape, batch_types.shape, batch_labels.shape, batch_neg.shape)
                if batch_inputs.shape[0]==0:
                    break
                context_loss, type_loss = self.model(batch_inputs, input_types, batch_labels, batch_types, batch_neg, batch_type_masks)
                total_loss = context_loss + type_loss
                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()
                step += 1

                avg_loss += total_loss.item()

                if step % 5000 == 0:
                    print('Epoch: {}, iter {}, total loss: {}, context loss: {}, type loss: {}'.format(epoch, step,
                                                                                                       total_loss.item(),
                                                                                                       context_loss.item(),
                                                                                                       type_loss.item()))
                    

            avg_loss /= step
            print("Epoch: {}, avg_loss={}".format(epoch, avg_loss))
        elif mode == 'train':
            while pipeline.sen_index != len(self.sentences) and step < 200000:
                # batch_inputs, batch_labels = pipeline.generate_batch_one(skip_window)
                batch_inputs, input_types, batch_types, batch_labels, batch_type_masks = pipeline.generate_batch(skip_window)
                batch_neg = pipeline.get_neg_data(num_neg, batch_inputs)

                batch_inputs = torch.LongTensor(batch_inputs).to(self.device)
                input_types = torch.LongTensor(input_types).to(self.device)
                batch_labels = torch.LongTensor(batch_labels).to(self.device)
                batch_neg = torch.LongTensor(batch_neg).to(self.device)
                batch_types = torch.Tensor(batch_types).to(self.device)
                batch_type_masks = torch.Tensor(batch_type_masks).to(self.device)
                # print(batch_inputs.shape, batch_labels.shape, batch_neg.shape)
                context_loss, type_loss = self.model(batch_inputs, input_types, batch_labels, batch_types, batch_neg, batch_type_masks)
                total_loss = context_loss + type_loss
                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()
                step += 1

                avg_loss += total_loss.item()

                if step % 5000 == 0:
                    print('Epoch: {}, iter {}, total loss: {}, context loss: {}, type loss: {}'.format(epoch, step,
                                                                                                       total_loss.item(),
                                                                                                       context_loss.item(),
                                                                                                       type_loss.item()))

            avg_loss /= step
            print("Epoch: {}, avg_loss={}".format(epoch, avg_loss))
        else:
            print("no such mode")
            exit()

        # save model at last
        # torch.save(self.model.state_dict(), self.outputdir + '/model_step%d.pt' % train_steps)

    def save_model(self, epoch, out_path):
        torch.save(self.model.state_dict(), out_path)

    def get_list_vector(self):
        sd = self.model.state_dict()
        return sd['input_emb.weight']

    def save_embed(self, epoch, path_dir='embedding_dw'):
        out_file = os.path.join(path_dir, "embedding_epoch{}".format(epoch))
        # out_file = "./dw_init_embed"

        embed_dw = gensim.models.keyedvectors.Word2VecKeyedVectors(self.args.embedding_size)
        embed = self.model.input_emb.weight.data.cpu().numpy()
        embed_dw.add(['PAD', 'MASK'] + self.args.nodes, embed)
        embed_dw.save_word2vec_format(fname=out_file, binary=False)

        return torch.from_numpy(embed).float()

    def load_embed(self):
        return self.model.input_emb.weight.data

    def load_model(self, model_path):
        checkpoint=torch.load(model_path, map_location=self.device)
        # ind_to_loc=self.vocab.index2loc
        # for i in range(self.vocab.specials_num):
        #     ind_to_loc[i]=checkpoint['input_emb.weight'].shape[0]-1
        # checkpoint['input_emb.weight']=checkpoint['input_emb.weight'][ind_to_loc,:]
        # checkpoint['output_emb.weight']=checkpoint['output_emb.weight'][ind_to_loc,:self.model.output_emb.weight.shape[1]]
        # checkpoint['type_pred.weight']=checkpoint['type_pred.weight'][:self.model.type_pred.weight.shape[0]]
        self.model.load_state_dict(checkpoint)
        # self.save_model(100,model_path)

    def vector(self, index):
        self.model.predict(index)

    def most_similar(self, word, top_k=8):
        index = self.word2index[word]
        index = torch.tensor(index, dtype=torch.long).cuda().unsqueeze(0)
        emb = self.model.predict(index)
        sim = torch.mm(emb, self.model.input_emb.weight.transpose(0, 1))
        nearest = (-sim[0]).sort()[1][1: top_k + 1]
        top_list = []
        for k in range(top_k):
            close_word = self.index2word[nearest[k].item()]
            top_list.append(close_word)
        return top_list
    

class BertModel4Pretrain(nn.Module):
    def __init__(self, config,data_feature):
        super(BertModel4Pretrain, self).__init__()
        dim=config.get('hidden_dim',64)
        layers=config.get('layers',4)
        n_vocab=data_feature.get('num_nodes')
        max_len=config.get('max_len')
        p_dropout=config.get('p_dropout',0.5)
        n_heads=config.get('n_heads')
        dim_ff=config.get('ff_dim',64)

        self.transformer = Transformer(dim,layers,n_vocab,dim,max_len,p_dropout,n_heads,dim_ff)
        self.hidden_dim=config.get('hidden_dim',64)
        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.activ2 = gelu
        self.norm = LayerNorm(dim)
        self.classifier = nn.Linear(self.hidden_dim, 2)

        embed_weight = self.transformer.embed.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab)


    def init_token_embed(self, embed):
        token_vocab = self.transformer.embed.tok_embed.weight.shape[0]

        if embed.shape[0] < token_vocab:
            self.transformer.embed.tok_embed.weight.data[token_vocab-embed.shape[0]:] = embed
            print(self.transformer.embed.tok_embed.weight.shape)
        else:
            self.transformer.embed.tok_embed.weight.data = embed

    def forward(self, input_ids, input_mask, masked_pos, traj_len):
        h = self.transformer(input_ids, input_mask) # B x S x D
        # pooled_h = self.activ1(self.fc(h[:, 0]))

        traj_h = torch.sum(h * input_mask.unsqueeze(-1).float(), dim = 1)/ traj_len.unsqueeze(-1).float()
        pooled_h = self.activ1(self.fc(traj_h))

        masked_pos = masked_pos.expand(-1, -1, h.size(-1)).long() # B x S x D
        h_masked = h*masked_pos
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        # logits_lm = self.decoder(h_masked) + self.decoder_bias
        logits_lm = self.decoder(h_masked)
        logits_clsf = self.classifier(pooled_h)

        return logits_lm, logits_clsf

    def encode_sequence(self, input_ids, input_mask):
        h = self.transformer(input_ids, input_mask) # B x S x D
        input_mask=input_mask.unsqueeze(-1).float()
        traj_h = torch.sum(h * input_mask, dim = 1)/ input_mask.sum(1).float() 
        return traj_h