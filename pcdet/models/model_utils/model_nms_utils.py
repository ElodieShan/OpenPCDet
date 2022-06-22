import torch

def class_score_filter(box_scores, box_preds, score_thresh=None):
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
    return original_idxs, src_box_scores[original_idxs]
