import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
import os
import cv2
import matplotlib.pyplot as plt
from loss import FocalLoss, SSIM  # Import loss functions used in training

def find_optimal_threshold(y_true, y_score):
    """
    Find optimal threshold for F1 score
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    if len(thresholds) > optimal_idx:
        return thresholds[optimal_idx]
    else:
        return 0.5  # Default if thresholds array is shorter than optimal_idx

def calculate_iou(pred_mask, true_mask):
    """
    Calculate Intersection over Union (IoU)
    """
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return 1.0  # If both masks are empty, IoU is 1
    return intersection / union

def visualize_results(obj_names, metrics, save_path='./metric_visualizations/'):
    """
    Visualize metrics across different object classes
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Extract metrics
    image_auc = [metrics[obj]['image_auc'] for obj in obj_names]
    pixel_auc = [metrics[obj]['pixel_auc'] for obj in obj_names]
    image_ap = [metrics[obj]['image_ap'] for obj in obj_names]
    pixel_ap = [metrics[obj]['pixel_ap'] for obj in obj_names]
    f1_pixel = [metrics[obj]['f1_pixel'] for obj in obj_names]
    iou_pixel = [metrics[obj]['iou_pixel'] for obj in obj_names]
    f1_image = [metrics[obj]['f1_image'] for obj in obj_names]
    iou_image = [metrics[obj]['iou_image'] for obj in obj_names]
    thresholds = [metrics[obj]['threshold'] for obj in obj_names]
    
    # Plot metrics
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.bar(obj_names, image_auc, color='blue', alpha=0.7)
    plt.bar(obj_names, pixel_auc, color='orange', alpha=0.7)
    plt.title('AUC Scores')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(['Image AUC', 'Pixel AUC'])
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(2, 2, 2)
    plt.bar(obj_names, image_ap, color='blue', alpha=0.7)
    plt.bar(obj_names, pixel_ap, color='orange', alpha=0.7)
    plt.title('AP Scores')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(['Image AP', 'Pixel AP'])
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(2, 2, 3)
    plt.bar(obj_names, f1_pixel, color='blue', alpha=0.7)
    plt.bar(obj_names, f1_image, color='orange', alpha=0.7)
    plt.title('F1 Scores')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(['Pixel F1', 'Image F1'])
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(2, 2, 4)
    plt.bar(obj_names, iou_pixel, color='blue', alpha=0.7)
    plt.bar(obj_names, iou_image, color='orange', alpha=0.7)
    plt.bar(obj_names, thresholds, color='green', alpha=0.5)
    plt.title('IoU Scores & Thresholds')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(['Pixel IoU', 'Image IoU', 'Threshold'])
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'metrics_comparison.png'))
    plt.close()
    
    print(f"Metrics visualization saved to {save_path}")

def save_sample_visualizations(image, gt_mask, pred_mask, score, obj_name, sample_idx, vis_path):
    """
    Save visualization of sample results
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(image)
    mask_vis = np.zeros_like(image)
    mask_vis[:,:,1] = gt_mask * 255  # Green channel for ground truth
    axes[1].imshow(mask_vis, alpha=0.5)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Predicted mask
    axes[2].imshow(image)
    mask_vis = np.zeros_like(image)
    mask_vis[:,:,0] = pred_mask * 255  # Red channel for prediction
    axes[2].imshow(mask_vis, alpha=0.5)
    axes[2].set_title(f'Prediction (Score: {score:.3f})')
    axes[2].axis('off')
    
    # Heatmap of anomaly scores
    im = axes[3].imshow(pred_mask, cmap='jet')
    axes[3].set_title('Anomaly Heatmap')
    axes[3].axis('off')
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_path, f'{obj_name}_sample_{sample_idx}.png'))
    plt.close()

def write_results_to_file(run_name, metrics, obj_names, output_path='./outputs/'):
    """
    Write detailed results to file
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create a formatted output string
    fin_str = f"Results for {run_name}\n"
    fin_str += "=" * 50 + "\n\n"
    
    # Add header row
    fin_str += f"{'Metric':<15}{'Mean':<10}"
    for obj in obj_names:
        fin_str += f"{obj:<15}"
    fin_str += "\n" + "-" * (15 + 10 + 15 * len(obj_names)) + "\n"
    
    # Add metric rows
    metrics_list = ['image_auc', 'pixel_auc', 'image_ap', 'pixel_ap', 
                   'f1_pixel', 'iou_pixel', 'f1_image', 'iou_image', 'threshold']
    
    metric_names = {
        'image_auc': 'Image AUC', 
        'pixel_auc': 'Pixel AUC', 
        'image_ap': 'Image AP', 
        'pixel_ap': 'Pixel AP',
        'f1_pixel': 'F1 Pixel', 
        'iou_pixel': 'IoU Pixel', 
        'f1_image': 'F1 Image', 
        'iou_image': 'IoU Image',
        'threshold': 'Threshold'
    }
    
    for metric in metrics_list:
        values = [metrics[obj][metric] for obj in obj_names]
        mean_val = np.mean(values)
        
        fin_str += f"{metric_names[metric]:<15}{mean_val:<10.3f}"
        for val in values:
            fin_str += f"{val:<15.3f}"
        fin_str += "\n"
    
    # Write to file
    with open(os.path.join(output_path, f"{run_name}_detailed_results.txt"), 'w') as file:
        file.write(fin_str)
    
    # Also append to the combined results file
    with open(os.path.join(output_path, "combined_results.txt"), 'a') as file:
        file.write(fin_str + "\n\n" + "=" * 50 + "\n\n")
    
    print(f"Detailed results written to {output_path}")

def test(obj_names, mvtec_path, checkpoint_path, base_model_name, visualize_masks=False, visualization_path='./visualizations/', 
         adaptive_threshold=True):
    """
    Test function with enhanced visualization and metrics calculation
    
    Args:
        obj_names: List of object names to test
        mvtec_path: Path to MVTec dataset
        checkpoint_path: Path to model checkpoints
        base_model_name: Base name of the model
        visualize_masks: Whether to visualize masks
        visualization_path: Path to save visualizations
        adaptive_threshold: Whether to use adaptive thresholding (find optimal threshold for each object)
    """
    # Import models
    from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
    
    if visualize_masks and not os.path.exists(visualization_path):
        os.makedirs(visualization_path)
    
    # Dictionary to store all metrics for visualization
    all_metrics = {}
    
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []
    obj_f1_pixel_list = []
    obj_iou_pixel_list = []
    obj_f1_image_list = []
    obj_iou_image_list = []
    obj_thresholds_list = []
    
    for obj_name in obj_names:
        print(f"\nProcessing {obj_name}...")
        img_dim = 256
        run_name = base_model_name+"_"+obj_name+'_'

        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model_path = os.path.join(checkpoint_path, run_name+".pckl")
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue
            
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        model.cuda()
        model.eval()

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        seg_model_path = os.path.join(checkpoint_path, run_name+"_seg.pckl")
        if not os.path.exists(seg_model_path):
            print(f"Segmentation model file not found: {seg_model_path}")
            continue
            
        model_seg.load_state_dict(torch.load(seg_model_path, map_location='cuda:0'))
        model_seg.cuda()
        model_seg.eval()

        dataset_path = mvtec_path + obj_name + "/test/"
        if not os.path.exists(dataset_path):
            print(f"Test dataset not found: {dataset_path}")
            continue
            
        dataset = MVTecDRAEMTestDataset(dataset_path, resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []
        
        # For pixel-level metrics
        all_binary_preds_pixel = []
        all_binary_gts_pixel = []
        all_pixel_scores = []
        all_pixel_gts = []
        
        # For image-level metrics
        image_level_pred = []
        image_level_gt = []
        
        # For visualization
        if visualize_masks:
            obj_vis_path = os.path.join(visualization_path, obj_name)
            if not os.path.exists(obj_vis_path):
                os.makedirs(obj_vis_path)
                
        # Sample images for visualization
        visualization_indices = np.random.choice(len(dataloader), min(10, len(dataloader)), replace=False)
        
        for i_batch, sample_batched in enumerate(dataloader):
            gray_batch = sample_batched["image"].cuda()

            is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

            with torch.no_grad():
                gray_rec = model(gray_batch)
                joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)
                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

            out_mask_cv = out_mask_sm[0, 1, :, :].detach().cpu().numpy()
            
            # Store for adaptive threshold calculation
            flat_true_mask = true_mask_cv.flatten()
            flat_out_mask = out_mask_cv.flatten()
            
            all_pixel_scores.extend(flat_out_mask)
            all_pixel_gts.extend(flat_true_mask)
            
            # Store for metrics calculation
            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
            
            # Calculate image score - use max of avg-pooled anomaly map
            out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:, 1:, :, :], 21, stride=1,
                                                             padding=21 // 2).cpu().detach().numpy()
            image_score = np.max(out_mask_averaged)
            anomaly_score_prediction.append(image_score)
            
            # Store the ground truth for image level (1 if has anomaly, 0 otherwise)
            image_level_gt.append(1 if np.any(true_mask_cv > 0) else 0)
            
            mask_cnt += 1
            
            # Visualize some samples
            if visualize_masks and i_batch in visualization_indices:
                # Get original image for visualization
                orig_img = gray_batch[0].detach().cpu().numpy().transpose(1, 2, 0)
                orig_img = (orig_img * 255).astype(np.uint8)
                
                # Save full visualization with original image, ground truth, prediction
                save_sample_visualizations(
                    orig_img, 
                    true_mask_cv.squeeze(), 
                    out_mask_cv, 
                    image_score,
                    obj_name, 
                    i_batch, 
                    obj_vis_path
                )
        
        # Calculate optimal threshold if adaptive_threshold is True
        if adaptive_threshold and len(all_pixel_gts) > 0 and sum(all_pixel_gts) > 0:
            optimal_threshold = find_optimal_threshold(all_pixel_gts, all_pixel_scores)
            print(f"Optimal threshold for {obj_name}: {optimal_threshold:.3f}")
        else:
            optimal_threshold = 0.2  # Default threshold
        
        obj_thresholds_list.append(optimal_threshold)
        
        # Apply threshold to get binary predictions
        binary_preds_pixel = (total_pixel_scores > optimal_threshold).astype(np.uint8)
        binary_gts_pixel = (total_gt_pixel_scores > 0).astype(np.uint8)
        
        # Calculate image-level binary predictions
        for score in anomaly_score_prediction:
            image_level_pred.append(1 if score > optimal_threshold else 0)
        
        # Calculate metrics
        # Pixel-level metrics
        auroc_pixel = roc_auc_score(binary_gts_pixel, total_pixel_scores)
        ap_pixel = average_precision_score(binary_gts_pixel, total_pixel_scores)
        f1_pixel = f1_score(binary_gts_pixel, binary_preds_pixel)
        
        # Calculate IoU for pixel-level
        intersection = np.logical_and(binary_preds_pixel, binary_gts_pixel).sum()
        union = np.logical_or(binary_preds_pixel, binary_gts_pixel).sum()
        iou_pixel = intersection / union if union > 0 else 1.0
        
        # Image-level metrics
        auroc_image = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        ap_image = average_precision_score(anomaly_score_gt, anomaly_score_prediction)
        f1_image = f1_score(image_level_gt, image_level_pred)
        
        # Calculate IoU for image-level
        image_level_gt_np = np.array(image_level_gt)
        image_level_pred_np = np.array(image_level_pred)
        
        true_positives = np.sum(np.logical_and(image_level_pred_np == 1, image_level_gt_np == 1))
        false_positives = np.sum(np.logical_and(image_level_pred_np == 1, image_level_gt_np == 0))
        false_negatives = np.sum(np.logical_and(image_level_pred_np == 0, image_level_gt_np == 1))
        
        image_iou = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 1.0
        
        # Store metrics
        obj_ap_pixel_list.append(ap_pixel)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc_image)
        obj_ap_image_list.append(ap_image)
        obj_f1_pixel_list.append(f1_pixel)
        obj_iou_pixel_list.append(iou_pixel)
        obj_f1_image_list.append(f1_image)
        obj_iou_image_list.append(image_iou)
        
        # Store metrics in dictionary for visualization
        all_metrics[obj_name] = {
            'image_auc': auroc_image,
            'pixel_auc': auroc_pixel,
            'image_ap': ap_image,
            'pixel_ap': ap_pixel,
            'f1_pixel': f1_pixel,
            'iou_pixel': iou_pixel,
            'f1_image': f1_image,
            'iou_image': image_iou,
            'threshold': optimal_threshold
        }
        
        # Print metrics
        print(f"\nResults for {obj_name}:")
        print(f"AUC Image:  {auroc_image:.3f}")
        print(f"AP Image:   {ap_image:.3f}")
        print(f"AUC Pixel:  {auroc_pixel:.3f}")
        print(f"AP Pixel:   {ap_pixel:.3f}")
        print(f"F1 Score (Pixel): {f1_pixel:.3f}")
        print(f"IoU Score (Pixel): {iou_pixel:.3f}")
        print(f"F1 Score (Image): {f1_image:.3f}")
        print(f"IoU Score (Image): {image_iou:.3f}")
        print(f"Threshold: {optimal_threshold:.3f}")
        print("==============================")

    # Print mean metrics
    print(f"\nSummary for {run_name}")
    print(f"AUC Image mean:  {np.mean(obj_auroc_image_list):.3f}")
    print(f"AP Image mean:   {np.mean(obj_ap_image_list):.3f}")
    print(f"AUC Pixel mean:  {np.mean(obj_auroc_pixel_list):.3f}")
    print(f"AP Pixel mean:   {np.mean(obj_ap_pixel_list):.3f}")
    print(f"F1 Score (Pixel) mean:  {np.mean(obj_f1_pixel_list):.3f}")
    print(f"IoU Score (Pixel) mean:  {np.mean(obj_iou_pixel_list):.3f}")
    print(f"F1 Score (Image) mean:  {np.mean(obj_f1_image_list):.3f}")
    print(f"IoU Score (Image) mean:  {np.mean(obj_iou_image_list):.3f}")
    print(f"Average threshold: {np.mean(obj_thresholds_list):.3f}")

    # Create visualization of metrics
    visualize_results(obj_names, all_metrics)
    
    # Write detailed results to file
    write_results_to_file(run_name, all_metrics, obj_names)

if __name__=="__main__":
    import argparse
    import cv2
    from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--base_model_name', action='store', type=str, required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--visualize_masks', action='store_true', help='Enable mask visualization')
    parser.add_argument('--visualization_path', action='store', type=str, default='./visualizations/', 
                        help='Path to save visualized masks')
    parser.add_argument('--adaptive_threshold', action='store_true', help='Use adaptive thresholding')

    args = parser.parse_args()

    obj_list = [
                'wood'
                ]

    with torch.cuda.device(args.gpu_id):
        test(obj_list, args.data_path, args.checkpoint_path, args.base_model_name, 
             args.visualize_masks, args.visualization_path, args.adaptive_threshold)