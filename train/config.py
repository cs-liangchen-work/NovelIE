bert_dir = ''

from transformers import AutoTokenizer 
tokenizer = AutoTokenizer.from_pretrained(bert_dir, do_lower_case=False)

task_ner_labels = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
}

task_rel_labels = {
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE'],
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
}


def get_labelmap(label_list):
    tag2idx = {tag: idx for idx, tag in enumerate(label_list)}
    idx2tag = {idx: tag for idx, tag in enumerate(label_list)}
    return tag2idx, idx2tag


if __name__ == '__main__':
    a, b = get_labelmap(task_ner_labels['ace04'])
    print(a)
    print(b)
