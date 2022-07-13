import random
import pickle


def compute_f1_scores(pred_pt, gold_pt):
    n_tp, n_gold, n_pred = 0, 0, 0
    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])
        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}
    return scores


def if_element_overlap(s1, s2):
    if s1 == "NULL" or s2 == "NULL":
        return True
    s1 = set(s1.split(' '))
    s2 = set(s2.split(' '))
    if s1.intersection(s2):
        return True
    else:
        return False


def if_tuple_overlap(t1, t2, elements):
    if t1 is None or t2 is None:
        return False
    output = False
    if elements == ["extraction", "extraction"]:
        if if_element_overlap(t1[0], t2[0]) and t1[1] == t2[1]:
            output = True
        if if_element_overlap(t1[1], t2[1]) and t1[0] == t2[0]:
            output = True
    if elements == ["extraction", "extraction", "unique_cls"]:
        if if_element_overlap(t1[0], t2[0]) and t1[1] == t2[1]:
            output = True
        if if_element_overlap(t1[1], t2[1]) and t1[0] == t2[0]:
            output = True
    if elements == ["multi_cls", "extraction", "unique_cls"]:
        if if_element_overlap(t1[1], t2[1]) and t1[0] == t2[0]:
            output = True
    if elements == ["multi_cls", "extraction", "extraction", "unique_cls"]:
        if t1[0] == t2[0] and if_element_overlap(t1[1], t2[1]) and t1[2] == t2[2]:
            output = True
        if t1[0] == t2[0] and t1[1] == t2[1] and if_element_overlap(t1[2], t2[2]):
            output = True
    if elements == ["extraction", "unique_cls"]:
        if if_element_overlap(t1[0], t2[0]):
            output = True
    if elements == ["extraction", "multi_cls", "extraction"]:
        if if_element_overlap(t1[0], t2[0]) and t1[1] == t2[1] and t1[2] == t2[2]:
            output = True
        if if_element_overlap(t1[2], t2[2]) and t1[1] == t2[1] and t1[0] == t2[0]:
            output = True
    return output


def get_preds(sents, raw_outputs, task_config):
    elements = task_config['elements']
    index_extr = [i for i, element in enumerate(elements) if element == 'extraction']
    index_cls = [i for i, element in enumerate(elements) if element != 'extraction']
    preds = []
    for sent, raw in zip(sents, raw_outputs):
        # if "Ray" in sent:
        #     print()
        pred = []
        t_hist = []
        for prob, seq in raw:
            keep = 1
            if len(seq.split(' | ')) != len(elements)+1:
                continue
            t = seq.split(' | ')[:-1]
            is_true = seq.split(' | ')[-1]
            for i in index_extr:
                if t[i] != 'NULL' and t[i] not in sent:
                    keep = 0
            if t in t_hist or is_true != 'true':
                keep = 0
            if keep:
                pred.append(tuple(t))
            t_hist.append(t)
        preds.append(pred)
    # remove_overlapping
    fixed_preds = []
    for pred in preds:
        fixed_pred = []
        for x in pred:
            if not any([if_tuple_overlap(x, y, elements) for y in fixed_pred]):
                fixed_pred.append(x)
        fixed_preds.append(fixed_pred)
    return fixed_preds


def convert_targets_to_tuples(targets):
    outputs = []
    for target in targets:
        output = []
        if target != '':
            for x in target.split('||||'):
                t = x.split(' | ')[:-1]
                t = tuple(t)
                output.append(t)
        outputs.append(output)
    return outputs


def get_final_results(sents, targets, raw_outputs, task_config):
    targets = convert_targets_to_tuples(targets)
    preds = get_preds(sents, raw_outputs, task_config)
    bad_recall, bad_precision = [], []
    for sent, target, pred, raw_output in zip(sents, targets, preds, raw_outputs):
        d = {"sent": sent, "target": target, "pred": pred, "raw_output": raw_output}
        if set(pred).difference(set(target)):
            bad_precision.append(d)
        if set(target).difference(set(pred)):
            bad_recall.append(d)
    random.shuffle(bad_recall)
    random.shuffle(bad_precision)
    bad_cases = {"bad_precision": bad_precision, "bad_recall": bad_recall}
    scores = compute_f1_scores(preds, targets)
    return scores, bad_cases


if __name__ == "__main__":
    from data_utils import get_task_config
    task_config = get_task_config('task_config.json', 'acos', 'laptop14')
    all_results = pickle.load(open(f'outputs/test_results.pickle', 'rb'))
    sents, targets, raw_outputs = zip(*all_results)
    # raw_outputs = [raw[:7] for raw in raw_outputs]
    # tmp = [(sent, target, raw) for sent, target, raw in zip(sents, targets, raw_outputs) if len(target.split('||||')) == 3 and len(sent.split(' '))<15]
    scores, bad_cases = get_final_results(sents, targets, raw_outputs, task_config)
    print()
