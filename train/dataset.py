import sys
import torch
import pickle
import random
import numpy as np

from tqdm import tqdm

from allennlp.common.util import pad_sequence_to_length
from allennlp.nn.util import get_mask_from_sequence_lengths

class Dataset(object):
    def __init__(self, batch_size, seq_len, dataset):
        super(Dataset, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.construct_index(dataset)

    def construct_index(self, dataset):
        self.dataset = dataset
        self.index_length = len(dataset)
        self.shuffle_list = list(range(0, self.index_length))

    def shuffle(self):
        random.shuffle(self.shuffle_list)

    def get_tqdm(self, device, shuffle=True):
        return tqdm(self.reader(device, shuffle), mininterval=2, total=self.index_length // self.batch_size, leave=False, file=sys.stdout, ncols=80)

    def reader(self, device, shuffle):
        cur_idx = 0
        while cur_idx < self.index_length:
            end_index = min(cur_idx + self.batch_size, self.index_length)
            batch = [self.dataset[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            cur_idx = end_index
            yield self.batchify(batch, device)
        if shuffle:
            self.shuffle()

    def batchify(self, batch, device):
        sub_word_lens = [len(tup[0]) for tup in batch]
        max_sub_word_len = self.seq_len
    
        original_sentence_len = [len(tup[4]) for tup in batch]
        max_original_sentence_len = self.seq_len

        data_x, data_span, data_y = list(), list(), list()
        apps = list()

        for data in batch:
            all_doc_tokens, joint_set_re, tok_to_orig_index, orig_to_tok_index, sentence, spans, elem = data

            # joint_set_re.append([0, 0, 0])
            data_x.append(all_doc_tokens)
            data_span.append(spans)
            data_y.append(joint_set_re)
            apps.append(data)


        f = torch.LongTensor

        data_x = list(map(lambda x: pad_sequence_to_length(x, max_sub_word_len), data_x))
        bert_mask = get_mask_from_sequence_lengths(f(sub_word_lens), max_sub_word_len)

        data_span_tensor = np.zeros((len(data_x), max_original_sentence_len, 2), dtype=int)
        for i in range(len(data_span)):
            temp = data_span[i]
            data_span_tensor[i, :min(len(temp), max_original_sentence_len), :] = temp

        targets = []
        for elem in data_y:
            target = {}
            labels = [x[2] for x in elem]
            starts = [x[0] for x in elem]
            ends = [x[1] for x in elem]
            target = {
                'labels': f(labels).to(device),
                'starts': f(starts).to(device),
                'ends': f(ends).to(device)
            }
            targets.append(target)

        return [f(data_x).to(device),  
                bert_mask.to(device),
                f(data_span_tensor).to(device),
                targets,
                apps]


if __name__ == "__main__":
    import pickle
    filename = 'data_base.pk'
    data = pickle.load(open(filename, 'rb'))
    train_data, dev_data, test_data = data

    print(test_data[0])
    
    train_dataset = Dataset(5, 200, test_data)

    for batch in train_dataset.reader('cpu', False):
        data_x, bert_mask, data_span, data_y, apps = batch
        print(data_x[0])
        print(bert_mask[0])
        print(data_span[0])
        print(data_y[0])
        break