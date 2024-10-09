import os

import torch
import pickle

from config import task_ner_labels
from util import *
from preprocess import *
from dataset import *

from model1 import SetIE

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def load_dataset():
    filename = 'data_large.pk'
    data = pickle.load(open(filename, 'rb'))
    train_data, dev_data, test_data = data

    train_data = list(filter(lambda x: len(x[0]) < 150, train_data))
    dev_data = list(filter(lambda x: len(x[0]) < 160, dev_data))
    test_data = list(filter(lambda x: len(x[0]) < 160, test_data))

    return train_data, dev_data, test_data



def evaluate(model, test_dataset, device):
    model.eval()
    
    result = []
    golden = []
    tok_to_orgs = []

    with torch.no_grad():
        for batch in test_dataset.get_tqdm(device, False):
            data_x, bert_mask, data_span, data_y, apps = batch
            outputs = model.predict(data_x, bert_mask, data_span, data_y)

            pred_logits = outputs['pred_logits'].cpu()
            start_softmax = outputs['pred_start'].cpu()
            end_softmax = outputs['pred_end'].cpu()

            for app in apps:
                tok_to_orgs.append(app[2])

            for elem in data_y:
                labels = elem['labels']
                starts = elem['starts']
                ends = elem['ends']

                temp = []
                for l, s, e in zip(labels, starts, ends):
                    temp.append([s.item(), e.item(), l.item()])
                golden.append(temp)


            for pred_logit, start, end in zip(pred_logits, start_softmax, end_softmax):
                predict_labels = torch.argmax(pred_logit, dim=1)
                predict_start  = torch.argmax(start, dim=1)
                predict_end  = torch.argmax(end, dim=1)
                temp = []
                for l, s, e in zip(predict_labels, predict_start, predict_end):
                    temp.append([s.item(), e.item(), l.item()])
                result.append(temp)

            # print(torch.argmax(outputs['pred_logits'][5], dim=1))
            # print(data_y[5]['labels'])

    n_correct = 0
    n_predict = 0
    n_gold = 0

    for pre, gold, tok_to_org in zip(result, golden, tok_to_orgs):
        pre = list(filter(lambda x: x[-1] != len(task_ner_labels['ace04']) and x[0] != 159 and x[1] != 159, pre))

        pre = list(filter(lambda x: -1 < x[-1] < 7 and 0 <= x[1] - x[0] < 5, pre))
        gold = list(filter(lambda x: -1 < x[-1] < 7, gold))

        # pre = list(filter(lambda x: -1 + 7 < x[-1] and 0 <= x[1] - x[0] < 5, pre))
        # gold = list(filter(lambda x: -1 + 7 < x[-1], gold))

        # pre = list(filter(lambda x: -1 + 7 + 18 < x[-1] < 7 + 18 + 33 and 0 <= x[1] - x[0] < 5, pre))
        # gold = list(filter(lambda x: -1 + 7 + 18 < x[-1] < 7 + 18 + 33, gold))

        # pre = list(filter(lambda x: -1 + 7 + 18 + 33 < x[-1], pre))
        # gold = list(filter(lambda x: -1 + 7 + 18 + 33 < x[-1], gold))

        ## ---
        # set_pre = set([(tok_to_org[x] if x < len(tok_to_org) else -1, tok_to_org[y] if y < len(tok_to_org) else -1, z) for x, y, z in pre])
        # set_gold = set([(tok_to_org[x] if x < len(tok_to_org) else -1, tok_to_org[y] if y < len(tok_to_org) else -1, z) for x, y, z in gold])

        set_pre = set([(x, y, z) for x, y, z in pre])
        set_gold = set([(x, y, z) for x, y, z in gold])

        print(set_pre)
        print(set_gold)
        print('----')

        n_gold += len(set_gold)
        n_predict += len(set_pre)
        n_correct += len(set_gold.intersection(set_pre))

    try:
        p = n_correct / n_predict
        r = n_correct / n_gold
        f1 = 2 * p * r / (p + r)
    except:
        p, r, f1 = 0, 0, 0
    print('P, R, F1', p, r, f1)
    return p, r, f1



if __name__ == '__main__':

    device = 'cuda'

    batch_size = 10
    train_data, dev_data, test_data = load_dataset()
    test_dataset = Dataset(batch_size, 160, test_data)   # <---- dev data

    model = SetIE(k_query=30, y_num=len(task_ner_labels['ace04']), y_len=160)
    model.to(device)
    # load_model(model, 'models/model_entity_event_15.pk')
    # load_model(model, 'models/model_entity_event_softmax_mask_3_19.pk')
    # load_model(model, 'models/model_entity_softmax_mask_3_19.pk')
    load_model(model, 'models/model_entity_softmax_mask_3_concate_13.pk')
    # load_model(model, 'models/model_entity_event_softmax_mask_1_bias_19.pk')


    evaluate(model, test_dataset, device)



        