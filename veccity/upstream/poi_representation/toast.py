import os
import random
import torch
from torch import optim
from torch import nn
from dataset import DataPipeline
from w2v import SkipGram
from TransEncoder import BERT


class ToastTrainer:
    def __init__(self, data_pipline, embedding_size, device='cuda', learning_rate=1.0):

        self.device = device

        # Data Pipline
        self.data_pipline: DataPipeline = data_pipline

        self.vocab_size = self.data_pipline.vocabulary_size
        self.traj_stats = self.data_pipline.data_traj_real_stats
        self.data, self.word_count, self.word2index, self.index2word = self.data_pipline.get_data_set()

        # Models
        self.sg_model: SkipGram = SkipGram(self.vocab_size, embedding_size, self.data_pipline.road_attrs_statis).to(
            device)
        self.sg_model_optim = optim.SGD(self.sg_model.parameters(), lr=learning_rate)

        self.tf_model: BERT = BERT(self.vocab_size, self.traj_stats['max'], embedding_size).to(device)
        self.tf_model_optim = optim.Adadelta(self.tf_model.parameters(), lr=learning_rate)

        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_epoch, skip_window=1, num_skips=2, batch_size=64, output_dir='out'):
        # self.outputdir = os.mkdir(output_dir)

        sg_avg_loss = 0
        tf_avg_loss = 0

        for step in range(train_epoch):

            # weight transport from transformer encoders embedding layer to skip-gram embedding layer
            tf_model_input_emb = self.tf_model.embedding.tok_embed.state_dict()
            self.sg_model.v_emb.load_state_dict(tf_model_input_emb, strict=True)

            # train skip_gram model
            batch_inputs, batch_labels, batch_attr = self.data_pipline.generate_sg_batch(batch_size, num_skips,
                                                                                         skip_window)

            batch_inputs = torch.tensor(batch_inputs, dtype=torch.long).to(self.device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(self.device)
            batch_attr = torch.tensor(batch_attr, dtype=torch.long).to(self.device)

            sg_loss = self.sg_model(batch_inputs, batch_labels, batch_attr)

            self.sg_model_optim.zero_grad()
            sg_loss.backward()
            self.sg_model_optim.step()

            # weight transport from skip-gram embedding layer to transformer encoders embedding layer
            sg_model_input_emb = self.sg_model.v_emb.state_dict()
            self.tf_model.embedding.tok_embed.load_state_dict(sg_model_input_emb, strict=True)

            # train transformer model
            input_ids, masked_tokens, masked_pos, isReal = self.data_pipline.generate_tf_batch(batch_size)

            input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
            masked_tokens = torch.tensor(masked_tokens, dtype=torch.long).to(self.device)
            masked_pos = torch.tensor(masked_pos, dtype=torch.long).to(self.device)
            isReal = torch.tensor(isReal, dtype=torch.long).to(self.device)

            logits_lm, logits_clsf = self.tf_model(input_ids, masked_pos)
            loss_lm = self.criterion(logits_lm.view(-1, self.vocab_size), masked_tokens.view(-1))  # for masked LM
            loss_lm = (loss_lm.float()).mean()
            loss_clsf = self.criterion(logits_clsf, isReal)  # for sentence classification
            tf_loss = loss_lm + loss_clsf

            self.tf_model_optim.zero_grad()
            tf_loss.backward()
            self.tf_model_optim.step()

            print("step ", step, " completed")

            sg_avg_loss += sg_loss.item()
            tf_avg_loss += tf_loss.item()
            if step % 20 == 0 and step > 0:
                sg_avg_loss /= 20
                tf_avg_loss /= 20
                print('Average sg loss at step ', step, ': ', sg_avg_loss)
                print('Average tf loss at step ', step, ': ', tf_avg_loss)
                sg_avg_loss = 0
                tf_avg_loss = 0

            # checkpoint
            if step % 100000 == 0 and step > 0:
                torch.save(self.sg_model.state_dict(), self.outputdir + '/model_step%d.pt' % step)

        # save model at last
        torch.save(self.sg_model.state_dict(), self.outputdir + '/model_step%d.pt' % train_epoch)

    def save_model(self, out_path):
        torch.save(self.sg_model.state_dict(), out_path + '/model.pt')

    def get_list_vector(self):
        sd = self.sg_model.state_dict()
        return sd['input_emb.weight'].tolist()

    def save_vector_txt(self, path_dir):
        embeddings = self.get_list_vector()
        fo = open(path_dir + '/vector.txt', 'w')
        for idx in range(len(embeddings)):
            word = self.index2word[idx]
            embed = embeddings[idx]
            embed_list = [str(i) for i in embed]
            line_str = ' '.join(embed_list)
            fo.write(word + ' ' + line_str + '\n')
        fo.close()

    def load_model(self, model_path):
        self.sg_model.load_state_dict(torch.load(model_path))

    def vector(self, index):
        self.sg_model.predict(index)

    def most_similar(self, word, top_k=8):
        index = self.word2index[word]
        index = torch.tensor(index, dtype=torch.long).cuda().unsqueeze(0)
        emb = self.sg_model.predict(index)
        sim = torch.mm(emb, self.sg_model.v_emb.weight.transpose(0, 1))
        nearest = (-sim[0]).sort()[1][1: top_k + 1]
        top_list = []
        for k in range(top_k):
            close_word = self.index2word[nearest[k].item()]
            top_list.append(close_word)
        return top_list
