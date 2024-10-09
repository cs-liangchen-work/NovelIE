import json
import pickle
import config

tokenizer = config.tokenizer
tag2idx, idx2tag = config.get_labelmap(config.task_ner_labels['ace04'])


def build_bert_example(context):
    tok_to_orig_index = []
    orig_to_tok_index = []
    spans = []
    all_doc_tokens = []

    for (i, token) in enumerate(context):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)

        s = len(all_doc_tokens)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
        e = len(all_doc_tokens) - 1
        spans.append([s, e])

    return tokenizer.convert_tokens_to_ids(all_doc_tokens), tok_to_orig_index, orig_to_tok_index, spans
    

def preprocess_bert(result):
    res = []
    for elem in result:
        sentence, entity_set, relation_set = elem[0], elem[1], elem[2]
        sentence = ['[CLS]'] + sentence
        all_doc_tokens, tok_to_orig_index, orig_to_tok_index, spans = build_bert_example(sentence)

        joint_set_re = []
        for el in entity_set:
            x, y, c = el
            if c in tag2idx:
                # joint_set_re.append([orig_to_tok_index[x] + 1, orig_to_tok_index[y] + 1, tag2idx[c]])   # +1 is for [CLS]
                joint_set_re.append([x + 1, y + 1, tag2idx[c]])
            else:
                print('Error!', c)

        res.append([all_doc_tokens, joint_set_re, tok_to_orig_index, orig_to_tok_index, sentence, spans, elem])
    return res


if __name__ == '__main__':
    from util import *

    result = read_data_entity_relation('entity_relation_data/ace05/json/train.json', False)
    train = preprocess_bert(result)

    result = read_data_entity_relation('entity_relation_data/ace05/json/dev.json', False)
    dev = preprocess_bert(result)

    result = read_data_entity_relation('entity_relation_data/ace05/json/test.json', False)
    test = preprocess_bert(result)

    for i in range(20):
        print(result[i])



    # result = read_data_entity_relation('data/ace_ner_relation/json/train.json')
    # print(result[0])
    # train = preprocess_bert(result)
    # print(train[0])

    # result = read_data_entity_relation('data/ace_ner_relation/json/dev.json')
    # dev = preprocess_bert(result)

    # result = read_data_entity_relation('data/ace_ner_relation/json/test.json')
    # test = preprocess_bert(result)
    # print(result[22])
    # print(test[22])

    data = [train, dev, test]
    with open('data_base.pk','wb') as f:
        pickle.dump(data, f)



    

