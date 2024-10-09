import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import BertModel

from allennlp.common.util import pad_sequence_to_length
from allennlp.nn.util import get_mask_from_sequence_lengths
from allennlp.nn.util import weighted_sum

import config
bert_dir = config.bert_dir

from matcher import HungarianMatcher

class SetIE(nn.Module):
    def __init__(self, k_query=20, y_num=39, y_len=35):
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_dir, output_hidden_states=True)
        
        self.queries_start = nn.ModuleList([nn.Linear(self.bert.config.hidden_size, 1) for _ in range(k_query)])
        self.queries_end = nn.ModuleList([nn.Linear(self.bert.config.hidden_size, 1) for _ in range(k_query)])

        self.num_classes = y_num
        self.y_len = y_len

        self.fc = nn.Linear(self.bert.config.hidden_size, y_num + 1)

        self.matcher = HungarianMatcher(1, 1, 1)

        self.dropout = nn.Dropout(0.5)
    
    def getName(self):
        return self.__class__.__name__

    def predict(self, data_x, bert_mask, data_y):
        outputs = self.bert(data_x, attention_mask=bert_mask)
        hidden_states = outputs[2][-1]

        # print(hidden_states.size())

        query_logits_start = [query_layer(hidden_states).squeeze(-1) for query_layer in self.queries_start]
        query_logits_end = [query_layer(hidden_states).squeeze(-1) for query_layer in self.queries_end]

        start_softmax = [torch.softmax(x, dim=-1) for x in query_logits_start]  ## masked softmax
        end_softmax = [torch.softmax(x, dim=-1) for x in query_logits_end]      ## masked softmax

        temp_start = [weighted_sum(hidden_states, x) for x in start_softmax]
        temp_end = [weighted_sum(hidden_states, x) for x in end_softmax] 

        temp_all = [self.fc(x + y) for x, y in zip(temp_start, temp_end)]

        pred_logits = torch.stack(temp_all).permute(1, 0, 2)
        start_softmax = torch.stack(start_softmax).permute(1, 0, 2)
        end_softmax = torch.stack(end_softmax).permute(1, 0, 2)

        # print(pred_logits.size())
        # print(pred_logits.size())
        # print(start_softmax.size())
        # print(end_softmax.size())

        outputs = {
            'pred_logits': pred_logits,
            'pred_start': start_softmax,
            'pred_end': end_softmax
        }
        return outputs

    def forward(self, data_x, bert_mask, data_y, weight1, weight2, weight3):

        outputs = self.bert(data_x, attention_mask=bert_mask)
        hidden_states = outputs[2][-1]
        # add decoder??


        query_logits_start = [query_layer(hidden_states).squeeze(-1) for query_layer in self.queries_start]
        # print(len(query_logits_start))  k_query的数目
        # print(query_logits_start[0].shape) torch.Size([5, 200])
        query_logits_end = [query_layer(hidden_states).squeeze(-1) for query_layer in self.queries_end]
        

        start_softmax = [torch.softmax(x, dim=-1) for x in query_logits_start]  ## masked softmax?
        end_softmax = [torch.softmax(x, dim=-1) for x in query_logits_end]      ## masked softmax?
        # print(len(start_softmax))    30
        # print(start_softmax[0].shape)    torch.Size([5, 200])    
        # 每一个query，  每一个句子，  每一个token作为开始/结束位置的softmax后的概率。    
        
        temp_start = [weighted_sum(hidden_states, x) for x in start_softmax]
        temp_end = [weighted_sum(hidden_states, x) for x in end_softmax] 
        # print(len(temp_start))  30
        # print(temp_start[0].shape)  torch.Size([5, 1024])  句向量表示。
        # print(hidden_states.shape)  torch.Size([5, 200, 1024])
        

        temp_all = [self.fc(x + y) for x, y in zip(temp_start, temp_end)]
        # print(temp_all[0].shape)   torch.Size([5, 82]) 分类器。
        pred_logits = torch.stack(temp_all).permute(1, 0, 2)
        # print(pred_logits.shape)  torch.Size([5, 30, 82])

        start_softmax = torch.stack(start_softmax).permute(1, 0, 2)
        end_softmax = torch.stack(end_softmax).permute(1, 0, 2)
        # print(start_softmax.shape)  torch.Size([5, 30, 200])
        # assert 1==2

        # print(pred_logits.size())
        # print(pred_logits.size())
        # print(start_softmax.size())
        # print(end_softmax.size())

        outputs = {
            'pred_logits': pred_logits,
            'pred_start': start_softmax,
            'pred_end': end_softmax
        }
        targets = data_y


        indices = self.matcher(outputs, targets)
        
        ### label loss
        src_logits = outputs['pred_logits']
        
        idx = self._get_src_permutation_idx(indices)    

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o

        empty_weight = torch.ones(self.num_classes + 1).to(src_logits.device)
        empty_weight[-1] = weight1
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
        
        
        ### start loss
        src_start = outputs['pred_start']
        # print(src_start)
        # assert 1==2
        idx = self._get_src_permutation_idx(indices)

        target_start_o = torch.cat([t["starts"][J] for t, (_, J) in zip(targets, indices)])
        target_start = torch.full(src_start.shape[:2], self.y_len - 1,
                                    dtype=torch.int64, device=src_start.device)

        target_start[idx] = target_start_o

        empty_weight = torch.ones(self.y_len).to(src_start.device)
        empty_weight[-1] = weight2

        # print(src_start.transpose(1, 2).shape)
        # print(target_start.shape)
        # print(empty_weight.shape)        
        loss_ce_start = F.cross_entropy(src_start.transpose(1, 2), target_start, empty_weight)
        

        ### end loss
        src_end = outputs['pred_end']
        idx = self._get_src_permutation_idx(indices)

        target_end_o = torch.cat([t["ends"][J] for t, (_, J) in zip(targets, indices)])
        target_end = torch.full(src_end.shape[:2], self.y_len - 1,
                                    dtype=torch.int64, device=src_end.device)

        target_end[idx] = target_end_o

        empty_weight = torch.ones(self.y_len).to(src_end.device)
        empty_weight[-1] = weight2  # 
        
        loss_ce_end = F.cross_entropy(src_end.transpose(1, 2), target_end, empty_weight)
        
        return loss_ce + loss_ce_start + loss_ce_end

   
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


if __name__ == '__main__':
    import pickle
    from dataset import *
    import A_data
    train_x, train_y, _, _, _, _, train_tok, _, _ = A_data.getdata('5')
    batch = 5
    model = SetIE(k_query=30, y_num=39)

    '''
    # train的测试：
    for i in range(0, len(train_x)//5 - 1):
        tem_train_x = torch.LongTensor(train_x[i * 5: (i + 1) * 5])
        tem_train_y = train_y[i * 5: (i + 1) * 5]
        print(tem_train_y)
        # print(tem_train_x)
        # assert 1==2
        loss = model(tem_train_x, tem_train_x > 0, tem_train_y)
        print(loss)
    assert 1==2
    '''
    
    # predict的测试：
    for i in range(0, len(train_x)//5 - 1):
        tem_train_x = torch.LongTensor(train_x[i * 5: (i + 1) * 5])
        tem_train_y = train_y[i * 5: (i + 1) * 5]
        # print(tem_train_y)
        # print(tem_train_x)
        # assert 1==2
        outputs = model.predict(tem_train_x, tem_train_x > 0, tem_train_y)
        print(outputs['pred_logits'].shape)
        print(outputs['pred_start'].shape)
        print(outputs['pred_end'].shape)
        assert 1==2
