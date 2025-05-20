from sklearn import metrics
from skimage import measure
import cv2
import numpy as np
import pandas as pd

def compute_best_pr_re(anomaly_ground_truth_labels, anomaly_prediction_weights):
    precision, recall, thresholds = metrics.precision_recall_curve(anomaly_ground_truth_labels, anomaly_prediction_weights)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    f1_scores = np.nan_to_num(f1_scores, nan=0.0)

    if len(f1_scores) == 0:
        return thresholds[0] if len(thresholds) > 0 else 0.5, 0.0, 0.0

    best_f1_idx = np.argmax(f1_scores)

    if best_f1_idx == 0 and len(precision) > len(thresholds):
        best_threshold = thresholds[0] if len(thresholds) > 0 else (anomaly_prediction_weights.min() if len(anomaly_prediction_weights) > 0 else 0.5)
    elif best_f1_idx > 0 and best_f1_idx <= len(thresholds):
        best_threshold = thresholds[best_f1_idx - 1]
    elif best_f1_idx > len(thresholds) and len(thresholds) > 0:
        best_threshold = thresholds[-1]
    elif len(thresholds) > 0:
        best_threshold = thresholds[min(best_f1_idx, len(thresholds)-1)]
    else:
        best_threshold = anomaly_prediction_weights.mean() if len(anomaly_prediction_weights) > 0 else 0.5

    best_precision = precision[best_f1_idx]
    best_recall = recall[best_f1_idx]
    print(f"Output from compute_best_pr_re -> Best Threshold: {best_threshold:.4f}, Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")
    return best_threshold, best_precision, best_recall

def compute_imagewise_retrieval_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels, path='training'):
    auroc = metrics.roc_auc_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    ap = 0. if path == 'training' else metrics.average_precision_score(anomaly_ground_truth_labels, anomaly_prediction_weights)

    precision, recall, thresholds = metrics.precision_recall_curve(anomaly_ground_truth_labels, anomaly_prediction_weights)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)
    f1_scores = np.nan_to_num(f1_scores, nan=0.0)

    if len(f1_scores) == 0:
        max_f1 = 0.0
        best_f1_thresh = 0.5
    else:
        best_f1_idx = np.argmax(f1_scores)
        max_f1 = f1_scores[best_f1_idx]
        if best_f1_idx == 0 and len(precision) > len(thresholds):
            best_f1_thresh = thresholds[0] if len(thresholds) > 0 else (anomaly_prediction_weights.min() if len(anomaly_prediction_weights) > 0 else 0.5)
        elif best_f1_idx > 0 and best_f1_idx <= len(thresholds):
            best_f1_thresh = thresholds[best_f1_idx - 1]
        elif best_f1_idx > len(thresholds) and len(thresholds) > 0:
            best_f1_thresh = thresholds[-1]
        elif len(thresholds) > 0:
            best_f1_thresh = thresholds[min(best_f1_idx, len(thresholds)-1)]
        else:
            best_f1_thresh = anomaly_prediction_weights.mean() if len(anomaly_prediction_weights) > 0 else 0.5

    return {"auroc": auroc, "ap": ap, "max_f1": max_f1, "f1_threshold": best_f1_thresh}

def compute_pixel_f1(ground_truth_flat_masks, anomaly_flat_segmentations_heatmap, threshold=0.5):
    predictions_binary = (anomaly_flat_segmentations_heatmap >= threshold).astype(np.uint8)
    if np.sum(ground_truth_flat_masks) == 0 and np.sum(predictions_binary) == 0:
        return 1.0
    return metrics.f1_score(ground_truth_flat_masks, predictions_binary, average='binary', zero_division=0)

def compute_pixel_iou(ground_truth_flat_masks, anomaly_flat_segmentations_heatmap, threshold=0.5):
    predictions_binary = (anomaly_flat_segmentations_heatmap >= threshold).astype(np.uint8)
    if np.sum(ground_truth_flat_masks) == 0 and np.sum(predictions_binary) == 0:
        return 1.0
    return metrics.jaccard_score(ground_truth_flat_masks, predictions_binary, average='binary', zero_division=0)

def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks, path='training', segmentation_threshold=0.5):
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_ground_truth_masks = (ground_truth_masks.ravel() > 0.5).astype(np.uint8)
    flat_anomaly_segmentations_heatmap = anomaly_segmentations.ravel()

    auroc = metrics.roc_auc_score(flat_ground_truth_masks, flat_anomaly_segmentations_heatmap)
    ap = 0. if path == 'training' else metrics.average_precision_score(flat_ground_truth_masks, flat_anomaly_segmentations_heatmap)

    pixel_f1 = compute_pixel_f1(flat_ground_truth_masks, flat_anomaly_segmentations_heatmap, threshold=segmentation_threshold)
    pixel_iou = compute_pixel_iou(flat_ground_truth_masks, flat_anomaly_segmentations_heatmap, threshold=segmentation_threshold)

    return {"auroc": auroc, "ap": ap, "f1": pixel_f1, "iou": pixel_iou}

def compute_pro(masks, amaps, num_th=200):
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    if amaps.size == 0 or masks.size == 0 or amaps.shape[0] != masks.shape[0]:
        return 0.0

    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th = amaps.min()
    max_th = amaps.max()
    delta = 0.1

    if max_th > min_th and num_th > 0:
        delta = (max_th - min_th) / num_th
    elif max_th == min_th:
        num_th = 1
    elif num_th == 0:
        return 0.0

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    if max_th == min_th:
        threshold_range = np.array([min_th])
    else:
        threshold_range = np.arange(min_th, max_th + delta, delta)
        if len(threshold_range) == 0:
            threshold_range = np.array([min_th, max_th])

    for th in threshold_range:
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros_for_this_threshold = []
        for binary_amap_single, mask_single in zip(binary_amaps, masks):
            binary_mask_for_props = (mask_single > 0.5).astype(np.uint8)
            binary_amap_dilated = cv2.dilate(binary_amap_single.astype(np.uint8), k)
            labeled_mask = measure.label(binary_mask_for_props)
            if labeled_mask.max() == 0:
                continue
            props = measure.regionprops(labeled_mask)
            for prop in props:
                minr, minc, maxr, maxc = prop.bbox
                cropped_pred = binary_amap_dilated[minr:maxr, minc:maxc]
                cropped_mask = binary_mask_for_props[minr:maxr, minc:maxc]
                intersection = np.logical_and(cropped_pred, cropped_mask).sum()
                pro = intersection / cropped_mask.sum() if cropped_mask.sum() != 0 else 0
                pros_for_this_threshold.append(pro)

        if len(pros_for_this_threshold) == 0:
            continue

        pro_mean = np.mean(pros_for_this_threshold)
        fpr = np.logical_and(binary_amaps == 1, masks == 0).sum() / (masks == 0).sum()
        df = pd.concat([df, pd.DataFrame([[pro_mean, fpr, th]], columns=["pro", "fpr", "threshold"])])

    if df.empty:
        return 0.0

    df = df.sort_values("fpr")
    return np.trapz(df["pro"], df["fpr"])
