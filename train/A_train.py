import torch
import pickle

from config import task_ner_labels
from util import *
from preprocess import *
from dataset import *

from A_model import SetIE

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def predict_span_f1(pred_logits, pred_start, pred_end, tok, target_y):
    # 都改成已经放入的是一个维度的吧。
    # target_y的格式：
    # {'labels': tensor([31, 38]), 'starts': tensor([ 5, 12]), 'ends': tensor([ 9, 12])
    # print(pred_logits.shape, pred_start.shape, pred_end.shape)
    # print(len(tok))
    # print(tok)
    # print(target_y)
    n1, n2, n3 = 0, 0, 0  # 正确， 多预测， 错误预测。
    # print(pred_logits.argmax(-1).squeeze().shape, pred_start.argmax(-1).squeeze().shape, pred_end.argmax(-1).squeeze().shape)
    # print(pred_logits.argmax(-1).squeeze(), pred_start.argmax(-1).squeeze(), pred_end.argmax(-1).squeeze())
    def get_real_ind(tok, z1, z2):
        if z2 >= sum(tok):
            return -1, -1
        rz1, rz2 = -1, -1
        for i in range(0, len(tok)):
            if z1>=sum(tok[:i]) and z1<sum(tok[:i+1]):
                rz1 = i
                break
        for i in range(0, len(tok)):
            if z2>=sum(tok[:i]) and z2<sum(tok[:i+1]):
                rz2 = i
                break
        return rz1, rz2
    
    set_pre = set()
    for label,z1,z2 in zip(pred_logits.argmax(-1).squeeze(), pred_start.argmax(-1).squeeze(), pred_end.argmax(-1).squeeze()):
        z1, z2 = get_real_ind(tok, z1, z2)
        if z1 == -1 or z2 == -1:
            continue
        z1, z2, label = z1, z2, label.cpu().item()
        if label >= 34 or label == 17 or label == 10:
            continue
        set_pre.add((z1, z2, label))
    # print(target_y)
    set_gold = set()
    for label, y, z in zip(target_y['labels'], target_y['starts'], target_y['ends']):
        z1, z2 = get_real_ind(tok, y, z)
        assert z1 != -1 and z2 != -1
        z1, z2, label = z1, z2, label.cpu().item()
        if label >= 34 or label == 17 or label == 10:
            continue
        # print(z1, z2, label)
        set_gold.add((z1, z2, label))
    
    # set_gold = set([(y.cpu().item(), z.cpu().item(), x.cpu().item()) for x, y, z in zip(target_y['labels'], target_y['starts'], target_y['ends'])])
    # print(set_golden)
    # assert 1==2
    # print(set_pre)
    # print(set_gold)
    return len(set_gold), len(set_pre), len(set_gold.intersection(set_pre))          
         
def predict_span_f1_nsd(pred_logits, pred_start, pred_end, tok, target_y):
    # 都改成已经放入的是一个维度的吧。
    # target_y的格式：
    # {'labels': tensor([31, 38]), 'starts': tensor([ 5, 12]), 'ends': tensor([ 9, 12])
    # print(pred_logits.shape, pred_start.shape, pred_end.shape)
    # print(len(tok))
    # print(tok)
    # print(target_y)
    n1, n2, n3 = 0, 0, 0  # 正确， 多预测， 错误预测。
    # print(pred_logits.argmax(-1).squeeze().shape, pred_start.argmax(-1).squeeze().shape, pred_end.argmax(-1).squeeze().shape)
    # print(pred_logits.argmax(-1).squeeze(), pred_start.argmax(-1).squeeze(), pred_end.argmax(-1).squeeze())
    def get_real_ind(tok, z1, z2):
        if z2 >= sum(tok):
            return -1, -1
        rz1, rz2 = -1, -1
        for i in range(0, len(tok)):
            if z1>=sum(tok[:i]) and z1<sum(tok[:i+1]):
                rz1 = i
                break
        for i in range(0, len(tok)):
            if z2>=sum(tok[:i]) and z2<sum(tok[:i+1]):
                rz2 = i
                break
        return rz1, rz2
    
    set_pre = set()
    set_pre_ind = set()
    for label,z1,z2 in zip(pred_logits.argmax(-1).squeeze(), pred_start.argmax(-1).squeeze(), pred_end.argmax(-1).squeeze()):
        z1, z2 = get_real_ind(tok, z1, z2)
        if z1 == -1 or z2 == -1 or z1 > z2:
            continue
        z1, z2, label = z1, z2, label.cpu().item()
        if label >= 34 or label == 10 or label == 17:
            continue
        set_pre_ind.add((z1, z2))
    for label,z1,z2 in zip(pred_logits.argmax(-1).squeeze(), pred_start.argmax(-1).squeeze(), pred_end.argmax(-1).squeeze()):
        z1, z2 = get_real_ind(tok, z1, z2)
        if z1 == -1 or z2 == -1 or z1 > z2:
            continue
        z1, z2, label = z1, z2, label.cpu().item()
        if (label >= 34 or label == 17 or label == 10):
            set_pre.add((z1, z2))
    # print(target_y)
    set_gold = set()
    for label, y, z in zip(target_y['labels'], target_y['starts'], target_y['ends']):
        z1, z2 = get_real_ind(tok, y, z)
        assert z1 != -1 and z2 != -1
        z1, z2, label = z1, z2, label.cpu().item()
        if label >= 34 or label == 17 or label == 10:        
            set_gold.add((z1, z2))
    
    # set_gold = set([(y.cpu().item(), z.cpu().item(), x.cpu().item()) for x, y, z in zip(target_y['labels'], target_y['starts'], target_y['ends'])])
    # print(set_golden)
    # assert 1==2
    # print(set_pre)
    # print(set_pre_ind)
    # print(set_gold)
    # assert 1==2
    return len(set_gold), len(set_pre), len(set_gold.intersection(set_pre))          
         
def predict_span_f1_nsd_p(pred_logits, pred_start, pred_end, tok, target_y, p):
    # 都改成已经放入的是一个维度的吧。
    # target_y的格式：
    # {'labels': tensor([31, 38]), 'starts': tensor([ 5, 12]), 'ends': tensor([ 9, 12])
    # print(pred_logits.shape, pred_start.shape, pred_end.shape)
    # print(len(tok))
    # print(tok)
    # print(target_y)
    n1, n2, n3 = 0, 0, 0  # 正确， 多预测， 错误预测。
    # print(pred_logits.argmax(-1).squeeze().shape, pred_start.argmax(-1).squeeze().shape, pred_end.argmax(-1).squeeze().shape)
    # print(pred_logits.argmax(-1).squeeze(), pred_start.argmax(-1).squeeze(), pred_end.argmax(-1).squeeze())
    def get_real_ind(tok, z1, z2):
        if z2 >= sum(tok):
            return -1, -1
        rz1, rz2 = -1, -1
        for i in range(0, len(tok)):
            if z1>=sum(tok[:i]) and z1<sum(tok[:i+1]):
                rz1 = i
                break
        for i in range(0, len(tok)):
            if z2>=sum(tok[:i]) and z2<sum(tok[:i+1]):
                rz2 = i
                break
        return rz1, rz2
    
    set_pre = set()
    softmax = torch.nn.Softmax()
    for label, z1, z2, p1, p2, p3  in zip(pred_logits.argmax(-1).squeeze(), pred_start.argmax(-1).squeeze(), pred_end.argmax(-1).squeeze(), pred_logits, pred_start, pred_end):
        max_label_p = max(softmax(p1).cpu().tolist())
        max_begin_p = max(softmax(p2).cpu().tolist())
        max_end_p = max(softmax(p3).cpu().tolist())
        if max_begin_p + max_end_p < p:
            continue
        z1, z2 = get_real_ind(tok, z1, z2)
        if z1 == -1 or z2 == -1 or z1 > z2:
            continue
        z1, z2, label = z1, z2, label.cpu().item()
        if (label >= 34 or label == 17 or label == 10) and z2 - z1 <= 5:
            set_pre.add((z1, z2))
    # print(target_y)
    set_gold = set()
    for label, y, z in zip(target_y['labels'], target_y['starts'], target_y['ends']):
        z1, z2 = get_real_ind(tok, y, z)
        assert z1 != -1 and z2 != -1
        z1, z2, label = z1, z2, label.cpu().item()
        if label >= 34 or label == 17 or label == 10:        
            set_gold.add((z1, z2))
    
    # set_gold = set([(y.cpu().item(), z.cpu().item(), x.cpu().item()) for x, y, z in zip(target_y['labels'], target_y['starts'], target_y['ends'])])
    # print(set_golden)
    # assert 1==2
    # print(set_pre)
    # print(set_pre_ind)
    # print(set_gold)
    # assert 1==2
    return len(set_gold), len(set_pre), len(set_gold.intersection(set_pre)) 

def test(dev_x, dev_y, dev_tok, model, batch):
    n_gold, n_predict, n_correct= 0, 0, 0
    model.eval()
    for i in range(0, len(dev_x)//batch - 1):
        tem_dev_x = torch.LongTensor(dev_x[i * batch: (i + 1) * batch])
        tem_dev_y = dev_y[i * batch: (i + 1) * batch]
        tem_tok = dev_tok[i * batch: (i + 1) * batch]

        with torch.no_grad():
            outputs = model.module.predict(tem_dev_x.cuda(), (tem_dev_x > 0).cuda(), tem_dev_y)
        for p1,p2,p3,tok,y in zip(outputs['pred_logits'], outputs['pred_start'], outputs['pred_end'], tem_tok, tem_dev_y):
            n1, n2, n3 = predict_span_f1(p1,p2,p3,tok,y)
            n_gold += n1
            n_predict += n2
            n_correct += n3
    try:
        p = n_correct / n_predict
        r = n_correct / n_gold
        f1 = 2 * p * r / (p + r)
    except:
        p, r, f1 = 0, 0, 0
    print('P, R, F1', p, r, f1)
    return f1

def test_nsd(dev_x, dev_y, dev_tok, model, batch):
    n_gold, n_predict, n_correct = 0, 0, 0
    model.eval()
    for i in range(0, len(dev_x)//batch - 1):
        tem_dev_x = torch.LongTensor(dev_x[i * batch: (i + 1) * batch])
        tem_dev_y = dev_y[i * batch: (i + 1) * batch]
        tem_tok = dev_tok[i * batch: (i + 1) * batch]

        with torch.no_grad():
            outputs = model.module.predict(tem_dev_x.cuda(), (tem_dev_x > 0).cuda(), tem_dev_y)
        for p1,p2,p3,tok,y in zip(outputs['pred_logits'], outputs['pred_start'], outputs['pred_end'], tem_tok, tem_dev_y):
            n1, n2, n3 = predict_span_f1_nsd(p1,p2,p3,tok,y)
            n_gold += n1
            n_predict += n2
            n_correct += n3
    try:
        p = n_correct / n_predict
        r = n_correct / n_gold
        f1 = 2 * p * r / (p + r)
    except:
        p, r, f1 = 0, 0, 0
    print(n_gold, n_predict, n_correct)
    print('P, R, F1', p, r, f1)
    return f1

def test_nsd_p(dev_x, dev_y, dev_tok, model, batch):
    max_f1 = 0.0
    for p_ in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11]:
        n_gold, n_predict, n_correct = 0, 0, 0
        model.eval()
        for i in range(0, len(dev_x)//batch - 1):
            tem_dev_x = torch.LongTensor(dev_x[i * batch: (i + 1) * batch])
            tem_dev_y = dev_y[i * batch: (i + 1) * batch]
            tem_tok = dev_tok[i * batch: (i + 1) * batch]

            with torch.no_grad():
                outputs = model.module.predict(tem_dev_x.cuda(), (tem_dev_x > 0).cuda(), tem_dev_y)
            for p1,p2,p3,tok,y in zip(outputs['pred_logits'], outputs['pred_start'], outputs['pred_end'], tem_tok, tem_dev_y):
                n1, n2, n3 = predict_span_f1_nsd_p(p1,p2,p3,tok,y,p_)
                n_gold += n1
                n_predict += n2
                n_correct += n3
        try:
            p = n_correct / n_predict
            r = n_correct / n_gold
            f1 = 2 * p * r / (p + r)
        except:
            p, r, f1 = 0, 0, 0
        print(p_)
        print(n_gold, n_predict, n_correct)
        print('P, R, F1', p, r, f1)
        max_f1 = max(max_f1, f1)
    return max_f1

if __name__ == '__main__':
    '''@nni.get_next_parameter()'''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight1', type=float, default=10)
    parser.add_argument('--weight2', type=float, default=10)
    parser.add_argument('--weight3', type=float, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--k_query', type=int, default=30)
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    
    
    device = 'cuda'

    lr = 1e-5
    n_epochs = 50

    model = SetIE(k_query=args.k_query, y_num=34)
    # model = SetIE(k_query, y_num=len(task_ner_labels['ace05']), y_len=160)

    model = torch.nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    import A_data
    train_x, train_y, dev_x, dev_y, test_x, test_y, train_tok, dev_tok, test_tok = A_data.getdata('15')
    batch = 16
    batch_size = 16


    num_warmup_steps = 0
    num_training_steps = n_epochs * (len(train_x) / batch_size)
    warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = AdamW(parameters, lr=lr, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler

    max_f1 = 0.0
    
    for epc in range(n_epochs):
        model.train()
        for i in range(0, len(train_x)//batch - 1):
            tem_train_x = torch.LongTensor(train_x[i * batch: (i + 1) * batch])
            tem_train_y = train_y[i * batch: (i + 1) * batch]

            loss = model(tem_train_x.cuda(), (tem_train_x > 0).cuda(), tem_train_y, args.weight1, args.weight2, args.weight3)
            # print(loss)
            # assert 1==2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            # print(i, 'loss')
        print('========================', epc)
        # dev：
        print('dev:')
        dev_f1 = test(dev_x, dev_y, dev_tok, model, batch)
        dev_f1_nsd = test_nsd(dev_x, dev_y, dev_tok, model, batch)
        print('test:')
        test_f1 = test(test_x, test_y, test_tok, model, batch)        
        test_f1_nsd = test_nsd(test_x, test_y, test_tok, model, batch)
        test_f1_nsd_p = test_nsd_p(test_x, test_y, test_tok, model, batch)
        max_f1 = max(max_f1, max(test_f1_nsd, test_f1_nsd_p))
        print('/n')
        print(max_f1)
        print('/n')
        # save_model(model, 'models/model_entity_%d.pk' % (epc))
        torch.save(model.module.state_dict(), './models/static_dict' + str(epc) + '_' + str(test_f1) + '.pkl')
