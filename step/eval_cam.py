# Original Code: https://github.com/jiwoon-ahn/irn

import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
import torch

n_class = 21
def total_confusion_to_class_confusion(data):

    confusion_c = np.zeros((n_class, 2, 2))
    for i in range(n_class):
        confusion_c[i, 0, 0] = data[i, i]
        confusion_c[i, 0, 1] = np.sum(data[i, :]) - data[i, i]
        confusion_c[i, 1, 0] = np.sum(data[:, i]) - data[i, i]
        confusion_c[i, 1, 1] = np.sum(data) - np.sum(data[i, :]) - np.sum(data[:, i]) + data[i, i]

    return confusion_c

def run(args):
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    # labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

    preds = []
    labels = []
    n_images = 0
    for i, id in enumerate(dataset.ids):
        n_images += 1
        cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
        cams = cam_dict['high_res']

        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]
        preds.append(cls_labels.copy())
        labels.append(dataset.get_example_by_keys(i, (1,))[0])

    confusion = calc_semantic_segmentation_confusion(preds, labels)

    confusion_np = np.array(confusion)
    confusion_c = total_confusion_to_class_confusion(confusion_np).astype(float)
    precision, recall = [], []
    for i in range(n_class):
        recall.append(confusion_c[i, 0, 0] / np.sum(confusion_c[i, 0, :]))
        precision.append(confusion_c[i, 0, 0] / np.sum(confusion_c[i, :, 0]))

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator
    print("\n")

    print("threshold:", args.cam_eval_thres, 'miou:', np.nanmean(iou), "i_imgs", n_images, "precision", np.mean(np.array(precision)), "recall", np.mean(np.array(recall)))
    print("\n")


    print(iou)
    print(precision)
    print(recall)
    return np.nanmean(iou), np.mean(np.array(precision)), np.mean(np.array(recall))