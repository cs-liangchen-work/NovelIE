from audioop import lin2lin
from re import I
from tracemalloc import start


def getdata(file_name):
    # train
    with open('../data/SnipsNSD' + file_name + '%/train/seq.in') as f:
        train_seq_in = [ '[CLS] ' + line.strip() for line in f.readlines()]
    with open('../data/SnipsNSD' + file_name + '%/train/seq.out') as f:
        train_seq_out = [ 'O ' + line.strip() for line in f.readlines()]
    # dev
    with open('../data/SnipsNSD' + file_name + '%/valid/seq.in') as f:
        dev_seq_in = [ '[CLS] ' + line.strip() for line in f.readlines()]
    with open('../data/SnipsNSD' + file_name + '%/valid/seq.out') as f:
        dev_seq_out = [ 'O ' + line.strip() for line in f.readlines()]
    # test
    with open('../data/SnipsNSD' + file_name + '%/test/seq.in') as f:
        test_seq_in = [ '[CLS] ' + line.strip() for line in f.readlines()]
    with open('../data/SnipsNSD' + file_name + '%/test/seq.out') as f:
        test_seq_out = [ 'O ' + line.strip() for line in f.readlines()]

    # print(train_seq_in[0])
    # print(train_seq_out[0])
    # assert 1==2


    bert_dir = '/home/jliu/data/BertModel/bert-large-cased'
    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(bert_dir, do_lower_case=False)
    

    train_len = len(train_seq_in)
    dev_len = len(dev_seq_in)
    test_len = len(test_seq_in)
    
    seq_in = train_seq_in + dev_seq_in + test_seq_in
    seq_out = train_seq_out + dev_seq_out + test_seq_out

    for t1,t2 in zip(seq_in, seq_out):
        assert len(t1.split()) == len(t2.split())

    # all_label:
    all_label_set = set()
    for t in seq_out:
        t = t.split()
        for t_ in t:
            if t_ == 'O':
                all_label_set.add(t_)
            else:
                all_label_set.add(t_[2:])

    all_label_list = list(all_label_set)
    label2id = {}
    id2label = {}
    for i,j in enumerate(all_label_list):
        label2id[j] = i
        id2label[i] = j
    print(len(all_label_list))
    # print(label2id)
    #print(id2label)
    
    all_label_list = ['music_item', 'state', 'genre', 'playlist', 'city', 'object_name', 'entity_name', 'poi', 'artist', 'playlist_owner', 'ns', 'rating_unit', 'object_select', 'party_size_description', 'object_part_of_series_type', 'restaurant_name', 'object_type', 'sort', 'service', 'track', 'movie_type', 'cuisine', 'restaurant_type', 'current_location', 'year', 'served_dish', 'geographic_poi', 'spatial_relation', 'condition_temperature', 'best_rating', 'rating_value', 'movie_name', 'condition_description', 'O', 'facility', 'album', 'location_name', 'country', 'object_location_type']
    label2id = {'location_name': 0, 'object_location_type': 1, 'best_rating': 2, 'rating_value': 3, 'music_item': 4, 'party_size_number': 5, 'object_part_of_series_type': 6, 'rating_unit': 7, 'city': 8, 'served_dish': 9, 'O': 10, 'playlist': 11, 'current_location': 12, 'playlist_owner': 13, 'album': 14, 'movie_type': 15, 'facility': 16, 'ns': 17, 'year': 18, 'geographic_poi': 19, 'service': 20, 'state': 21, 'movie_name': 22, 'track': 23, 'object_name': 24, 'condition_temperature': 25, 'timeRange': 26, 'restaurant_type': 27, 'genre': 28, 'restaurant_name': 29, 'object_type': 30, 'cuisine': 31, 'party_size_description': 32, 'sort': 33}

    id2label = {0: 'location_name', 1: 'object_location_type', 2: 'best_rating', 3: 'rating_value', 4: 'music_item', 5: 'party_size_number', 6: 'object_part_of_series_type', 7: 'rating_unit', 8: 'city', 9: 'served_dish', 10: 'O', 11: 'playlist', 12: 'current_location', 13: 'playlist_owner', 14: 'album', 15: 'movie_type', 16: 'facility', 17: 'ns', 18: 'year', 19: 'geographic_poi', 20: 'service', 21: 'state', 22: 'movie_name', 23: 'track', 24: 'object_name', 25: 'condition_temperature', 26: 'timeRange', 27: 'restaurant_type', 28: 'genre', 29: 'restaurant_name', 30: 'object_type', 31: 'cuisine', 32: 'party_size_description', 33: 'sort'}    


    

    # 获取输入tokenizer之后的。
    train_feature = [tokenizer.tokenize(line) for line in seq_in]
    train_feature_id = [tokenizer.convert_tokens_to_ids(line) for line in train_feature]

    max_padding_len = 35
    for j in range(len(train_feature_id)):
        i = train_feature_id[j]
        if len(i) < max_padding_len:
            train_feature_id[j].extend([0] * (max_padding_len - len(i)))
        else:
            train_feature_id[j] = train_feature_id[j][0:max_padding_len - 1] + [train_feature_id[j][-1]]

    # 获取tokenizer_num的值。
    tokenizer_num = []
    for line in seq_in:
        s1 = line.split()
        s3 = tokenizer.tokenize(line)
        seq_token_len = []
        for word in s1:
            ss = tokenizer.tokenize(word)
            seq_token_len.append(len(ss))
        assert sum(seq_token_len) == len(s3)
        assert len(seq_token_len) == len(s1)
        tokenizer_num.append(seq_token_len)
    # print(len(tokenizer_num))
    # print(tokenizer_num[0])
    # assert 1==2
    def convert_real_id_2_tok_id(tok,begin,end):
        real_begin = sum(tok[:begin])
        real_end = sum(tok[:end + 1]) - 1
        # [1, 2, 5, 10]
        #  1 2 2 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 
        # [0, 1, 3, 8]
        return real_begin, real_end
    

    data_y = []
    data_x = []
    
    for seq, data, label, tok in zip(seq_in, train_feature_id, seq_out, tokenizer_num):
        # print('seq : ', seq)
        # print('data : ', data)
        # print('tok : ', tok)
        label = label.split()
        # print('label : ', label)
        
        labels, starts, ends = [], [], []
        for i,label_ in enumerate(label):
            # print(label_)
            if label_[:1] != 'B':
                continue
            if i == len(label) - 1:
                # print(i, i)
                begin, end = convert_real_id_2_tok_id(tok,i,i)
                # 加入集合
                starts.append(begin)
                ends.append(end)
                labels.append(label2id[label[i][2:]])
                continue
            for j in range(i+1, len(label)):
                if label[j][:1] != 'I':
                    # print(i, j-1)
                    begin, end = convert_real_id_2_tok_id(tok,i,j-1)
                    # 加入集合
                    # labels
                    # starts
                    # ends
                    starts.append(begin)
                    ends.append(end)
                    labels.append(label2id[label[i][2:]])
                    break
                if j == len(label) -1:
                    # print(i, j)
                    begin, end = convert_real_id_2_tok_id(tok,i,j)
                    starts.append(begin)
                    ends.append(end)
                    labels.append(label2id[label[i][2:]])
                    break
        import torch
        labels, starts, ends = torch.LongTensor(labels), torch.LongTensor(starts), torch.LongTensor(ends)
        tem_map = {}
        tem_map['labels'] = labels
        tem_map['starts'] = starts
        tem_map['ends'] = ends
        # print(tem_map)
        data_y.append(tem_map)
        data_x.append(data)
    # print(len(data_y))
    # print(train_len + dev_len + test_len)
    # print(len(data_x))
    # print(data_x[0])
    # print(train_len + dev_len + test_len)
    

    train_data_x, train_data_y, train_tok = data_x[:train_len], data_y[:train_len], tokenizer_num[:train_len]
    dev_data_x, dev_data_y, dev_tok = data_x[train_len:train_len+dev_len], data_y[train_len:train_len+dev_len], tokenizer_num[train_len:train_len+dev_len]
    test_data_x, test_data_y, test_tok = data_x[train_len+dev_len:], data_y[train_len+dev_len:], tokenizer_num[train_len+dev_len:]
    assert len(train_data_x) == len(train_data_y) and len(train_data_x) == train_len
    assert len(test_data_x) == len(test_data_y) and len(test_data_x) == test_len
    assert len(dev_data_x) == len(dev_data_y) and len(dev_data_x) == dev_len
    return train_data_x, train_data_y, dev_data_x, dev_data_y, test_data_x, test_data_y, train_tok, dev_tok, test_tok
    
if __name__ == '__main__':
    getdata('15')
