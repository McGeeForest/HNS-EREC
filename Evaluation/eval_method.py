import re
import numpy as np


def average_precision(gt, pred):
    """
    Computes the average precision.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    gt: set
             A set of ground-truth elements (order doesn't matter)
    pred: list
                A list of predicted elements (order does matter)

    Returns
    -------
    score: double
            The average precision over the input lists
    """

    if not gt:
        return 0.0

    score = 0.0
    num_hits = 0.0
    for i,p in enumerate(pred):
        if p in gt and p not in pred[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / max(1.0, len(gt))


def NDCG(gt, pred, use_graded_scores=False):
    score = 0.0
    for rank, item in enumerate(pred):
        if item in gt:
            if use_graded_scores:
                grade = 1.0 / (gt.index(item) + 1)
            else:
                grade = 1.0
            score += grade / np.log2(rank + 2)

    norm = 0.0
    for rank in range(len(gt)):
        if use_graded_scores:
            grade = 1.0 / (rank + 1)
        else:
            grade = 1.0
        norm += grade / np.log2(rank + 2)
    return score / max(0.3, norm)


def metrics(gt, pred, metrics_map):
    '''
    Returns a numpy array containing metrics specified by metrics_map.
    gt: set
            A set of ground-truth elements (order doesn't matter)
    pred: list
            A list of predicted elements (order does matter)
    '''
    out = np.zeros((len(metrics_map),), np.float32)

    if ('MAP' in metrics_map):
        avg_precision = average_precision(gt=gt, pred=pred)
        out[metrics_map.index('MAP')] = avg_precision

    if ('RPrec' in metrics_map):
        intersec = len(gt & set(pred[:len(gt)]))
        out[metrics_map.index('RPrec')] = intersec / max(1., float(len(gt)))

    if 'MRR' in metrics_map:
        score = 0.0
        for rank, item in enumerate(pred):
            if item in gt:
                score = 1.0 / (rank + 1.0)
                break
        out[metrics_map.index('MRR')] = score

    if ('NDCG' in metrics_map):
        out[metrics_map.index('NDCG')] = NDCG(gt, pred)

    for eval_item in metrics_map:
        if str(eval_item).__contains__('MRR@'):
            # top_select = int(eval_item.split("@")[1])
            top_select = int(str(eval_item).replace("MRR@", ""))
            print("top_select:", top_select)
            print("预测排名:", pred)
            print("真实排名:", gt)
            print("      值:", MRR_METHORD(top_select, pred, gt))
            out[metrics_map.index(eval_item)] = MRR_METHORD(top_select, pred, gt)
        if str(eval_item).__contains__('TOP@'):
            # top_select = int(eval_item.split("@")[1])
            top_select = int(str(eval_item).replace("TOP@", ""))
            print("top_select:", top_select)
            print("预测排名:", pred)
            print("真实排名:", gt)
            print("      值:", TOP_METHORD(top_select, pred, gt))
            out[metrics_map.index(eval_item)] = TOP_METHORD(top_select, pred, gt)
        print("===================")
    return out

    # # 计算MRR@10 - MRR@100的结果
    # if 'MRR@' in metrics_map:
    #     score = 0.0
    #     top_select = [i*10 for i in range(1, 11)]
    #     for top in top_select:
    #         for rank, item in enumerate(pred[:top]):
    #             if item in gt:
    #                 score = 1.0 / (rank + 1.0)
    #                 break
    #         out[metrics_map.index('MRR@'+str(top))] = score

    # # 计算TOP@10 - TOP@100的结果
    # if ('TOP@' in metrics_map):
    #     top_select = [i*10 for i in range(1, 11)]
    #     for top in top_select:
    #         pred = pred[:top]
    #         gt = gt[:1]
    #         topN = 0
    #         for item in pred:
    #             if item in gt:
    #                 topN += 1
    #         out[metrics_map.index('TOP@')+str(top)] = topN
    # return out

def MRR_METHORD(top_select, pred, gt):
    score = 0.0
    gt = gt[:1]
    for rank, item in enumerate(pred[:top_select]):
        if item in gt:
            score = 1.0 / (rank + 1.0)
            break
    return score
def TOP_METHORD(top_select, pred, gt):
    pred = pred[:top_select]
    gt = gt[:1]
    topN = 0
    for item in pred:
        if item in gt:
            topN += 1
    return topN