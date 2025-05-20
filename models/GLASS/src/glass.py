from loss import FocalLoss
from collections import OrderedDict
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Projection, PatchMaker

import numpy as np
import pandas as pd
import torch.nn.functional as F

import logging
import os
import math
import torch
import tqdm
import common
import metrics
import cv2
import utils
import glob
import shutil

LOGGER = logging.getLogger(__name__)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TBWrapper:
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1


class GLASS(torch.nn.Module):
    def __init__(self, device):
        super(GLASS, self).__init__()
        self.device = device
        self.input_shape = None
        self.pixel_metrics_threshold = 0.5
        self.current_epoch_for_vis = 0

    def load(
            self,
            backbone,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension,
            target_embed_dimension,
            patchsize=3,
            patchstride=1,
            meta_epochs=640,
            eval_epochs=1,
            dsc_layers=2,
            dsc_hidden=1024,
            dsc_margin=0.5,
            train_backbone=False,
            pre_proj=1,
            mining=1,
            noise=0.015,
            radius=0.75,
            p=0.5,
            lr=0.0001,
            svd=0,
            step=20,
            limit=392,
            **kwargs,
    ):

        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device

        self.forward_modules = torch.nn.ModuleDict({})
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(self.input_shape) 
        self.forward_modules["feature_aggregator"] = feature_aggregator

        self.preprocessing = common.Preprocessing(feature_dimensions, pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = self.preprocessing 

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(target_dim=self.target_embed_dimension)
        preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.meta_epochs = meta_epochs
        self.lr = lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.Adam(self.pre_projection.parameters(), lr, weight_decay=1e-5)

        self.eval_epochs = eval_epochs
        self.dsc_layers = dsc_layers
        self.dsc_hidden = dsc_hidden
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.AdamW(self.discriminator.parameters(), lr=lr * 2)
        self.dsc_margin = dsc_margin

        self.c = torch.tensor(0.0, device=self.device)
        self.p_val = p
        self.radius = radius
        self.mining = mining
        self.noise_val = noise
        self.svd = svd
        self.step_val = step
        self.limit = limit
        self.distribution = 0
        self.focal_loss = FocalLoss()

        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        if self.input_shape and len(self.input_shape) == 3:
            rescale_target_size = self.input_shape[-2:] # (H, W)
        else:
            rescale_target_size = (256, 256)

        self.anomaly_segmentor = common.RescaleSegmentor(device=self.device, target_size=rescale_target_size)

        self.model_dir = ""
        self.dataset_name = ""
        self.logger = None

    def set_model_dir(self, model_dir, dataset_name):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir)

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features_from_backbone = self.forward_modules["feature_aggregator"](images, eval=False)
        else:
            self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features_from_backbone = self.forward_modules["feature_aggregator"](images, eval=True)

        raw_features_list = [features_from_backbone[layer] for layer in self.layers_to_extract_from]

        patch_data_list = []
        for feat_map in raw_features_list:
            current_feat_map = feat_map
            if current_feat_map.ndim == 3 and current_feat_map.shape[0] > 0 :
                B, L_seq, C_seq = current_feat_map.shape
                side_len = int(math.sqrt(L_seq))
                if side_len * side_len != L_seq:
                    continue
                current_feat_map = current_feat_map.view(B, side_len, side_len, C_seq).permute(0, 3, 1, 2)
            elif current_feat_map.ndim != 4:
                continue

            current_patches, current_shape_info = self.patch_maker.patchify(current_feat_map, return_spatial_info=True)
            patch_data_list.append({'patches': current_patches, 'shape_info': current_shape_info})

        if not patch_data_list:
            return torch.empty(0, device=self.device), [] if provide_patch_shapes else torch.empty(0, device=self.device)

        ref_Hp, ref_Wp = patch_data_list[0]['shape_info']

        embeddings_for_preprocessing = []

        for i, data_item in enumerate(patch_data_list):
            current_patches_tensor = data_item['patches'] 
            Hp_current, Wp_current = data_item['shape_info']
            B, L_current, C_feat, ps_h, ps_w = current_patches_tensor.shape

            flat_patch_embeddings = current_patches_tensor.reshape(B, L_current, -1)

            if i == 0: 
                reshaped_for_mm = flat_patch_embeddings.reshape(-1, flat_patch_embeddings.shape[-1])
                embeddings_for_preprocessing.append(reshaped_for_mm)
            else:
                x_for_interp = flat_patch_embeddings.permute(0, 2, 1).reshape(B, C_feat * ps_h * ps_w, Hp_current, Wp_current)
                x_interpolated = F.interpolate(
                    x_for_interp,
                    size=(ref_Hp, ref_Wp),
                    mode="bilinear",
                    align_corners=False
                )

                reshaped_for_mm = x_interpolated.permute(0, 2, 3, 1).reshape(B * ref_Hp * ref_Wp, -1)
                embeddings_for_preprocessing.append(reshaped_for_mm)

        output_of_preprocessing = self.forward_modules["preprocessing"](embeddings_for_preprocessing)

        patch_features_aggregated = self.forward_modules["preadapt_aggregator"](output_of_preprocessing)

        original_patch_shapes_to_return = [item['shape_info'] for item in patch_data_list]

        if provide_patch_shapes:
            return patch_features_aggregated, original_patch_shapes_to_return
        return patch_features_aggregated

    def trainer(self, training_data, val_data, name):
        state_dict = {}
        glob_pattern_trainer = os.path.join(self.ckpt_dir, 'ckpt_best*')
        ckpt_path_found = glob.glob(glob_pattern_trainer)
        ckpt_path_save = os.path.join(self.ckpt_dir, "ckpt.pth")

        if len(ckpt_path_found) != 0:
            LOGGER.info("Start testing, ckpt file found!")
            return [0.0] * 8 + [-1]

        self.distribution = getattr(training_data.dataset, 'distribution', 0)
        xlsx_path = './datasets/excel/' + name.split('_')[0] + '_distribution.xlsx'
        if self.distribution == 1:
            self.svd = 1
        elif self.distribution == 2:
            self.svd = 0
        elif self.distribution == 3:
            self.svd = 1
        elif self.distribution == 4 and os.path.exists(xlsx_path):
            df = pd.read_excel(xlsx_path)
            self.svd = 1 - df.loc[df['Class'] == name, 'Distribution'].values[0]
        elif os.path.exists(xlsx_path) and self.distribution != 1:
            df = pd.read_excel(xlsx_path)
            self.svd = df.loc[df['Class'] == name, 'Distribution'].values[0]
        elif self.distribution != 1:
            pass  

        if self.distribution == 1 and not ckpt_path_found:
            self.forward_modules.eval()
            if hasattr(self, 'pre_projection') and self.pre_proj > 0:
                self.pre_projection.eval()
            with torch.no_grad():
                img_means = []
                for i_c_dist, data_c_dist in enumerate(training_data):
                    if i_c_dist > 5 and self.limit > 20:
                        break
                    img_c_dist = data_c_dist["image"].to(torch.float).to(self.device)
                    img_means.append(torch.mean(img_c_dist, dim=0, keepdim=True))
                if img_means:
                    self.c_img_mean = torch.mean(torch.cat(img_means, dim=0), dim=0)
                    avg_img_np = utils.torch_format_2_numpy_img(self.c_img_mean.detach().cpu().numpy())
                    self.svd = utils.distribution_judge(avg_img_np, name)
                    avg_img_save_path = os.path.join(self.ckpt_dir, f'avg_img_svd{self.svd}_{name}.png')
                    os.makedirs(os.path.dirname(avg_img_save_path), exist_ok=True)
                    cv2.imwrite(avg_img_save_path, avg_img_np)
                else:
                    self.svd = 0
            return self.svd

        pbar = tqdm.tqdm(range(self.meta_epochs), unit='epoch', desc=f"Training {name}")
        pbar_str_val_metrics = ""
        num_metrics = 8
        best_record = None
        ckpt_path_best_file = ""

        def update_state_dict_local():
            state_dict["discriminator"] = OrderedDict({k: v.detach().cpu() for k, v in self.discriminator.state_dict().items()})
            if hasattr(self, 'pre_projection') and self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({k: v.detach().cpu() for k, v in self.pre_projection.state_dict().items()})
            if self.train_backbone and hasattr(self, 'backbone_opt'): 
                 state_dict["backbone"] = OrderedDict({k:v.detach().cpu() for k,v in self.backbone.state_dict().items()})


        for i_epoch in pbar:
            self.current_epoch_for_vis = i_epoch
            self.forward_modules.eval()
            if hasattr(self, 'pre_projection') and self.pre_proj > 0: self.pre_projection.eval()

            with torch.no_grad():
                all_outputs_for_c = []
                current_batch_size = training_data.batch_size if hasattr(training_data, 'batch_size') else 1
                num_samples_for_c = 0
                for i_c, data_c in enumerate(training_data):
                    if self.limit > 0 and num_samples_for_c >= self.limit : break
                    img_c = data_c["image"].to(torch.float).to(self.device)
                    current_embeds, _ = self._embed(img_c, evaluation=True, provide_patch_shapes=True)
                    if current_embeds.numel() == 0: continue
                    if hasattr(self, 'pre_projection') and self.pre_proj > 0:
                        current_embeds_proj = self.pre_projection(current_embeds)
                        current_embeds = current_embeds_proj[0] if isinstance(current_embeds_proj, tuple) and len(current_embeds_proj)==2 else current_embeds_proj
                    all_outputs_for_c.append(current_embeds)
                    num_samples_for_c += img_c.shape[0]

                if all_outputs_for_c:
                    concatenated_outputs = torch.cat(all_outputs_for_c, dim=0)
                    if concatenated_outputs.numel() > 0: self.c = torch.mean(concatenated_outputs, dim=0)
                    else: self.c = torch.zeros(self.target_embed_dimension, device=self.device)
                else: self.c = torch.zeros(self.target_embed_dimension, device=self.device)

            pbar_str_train_metrics, _, _ = self._train_discriminator(training_data, i_epoch, pbar, pbar_str_val_metrics)
            update_state_dict_local()

            if (i_epoch + 1) % self.eval_epochs == 0:
                images_val, scores_val, segmentations_val, labels_gt_val, masks_gt_val = self.predict(val_data)
                eval_results = self._evaluate(images_val, scores_val, segmentations_val, labels_gt_val, masks_gt_val, name, path='validation')
                i_auroc, i_ap, i_f1, p_auroc, p_ap, pixel_f1_m, pixel_iou_m, p_pro = eval_results

                log_name_prefix = f"{name}/val"
                self.logger.logger.add_scalar(f"{log_name_prefix}/i-auroc", i_auroc, i_epoch)
                self.logger.logger.add_scalar(f"{log_name_prefix}/i-ap", i_ap, i_epoch)
                self.logger.logger.add_scalar(f"{log_name_prefix}/i-f1", i_f1, i_epoch)
                self.logger.logger.add_scalar(f"{log_name_prefix}/p-auroc", p_auroc, i_epoch)
                self.logger.logger.add_scalar(f"{log_name_prefix}/p-ap", p_ap, i_epoch)
                self.logger.logger.add_scalar(f"{log_name_prefix}/p-f1", pixel_f1_m, i_epoch)
                self.logger.logger.add_scalar(f"{log_name_prefix}/p-iou", pixel_iou_m, i_epoch)
                self.logger.logger.add_scalar(f"{log_name_prefix}/p-pro", p_pro, i_epoch)

                current_eval_values = [i_auroc, i_ap, i_f1, p_auroc, p_ap, pixel_f1_m, pixel_iou_m, p_pro, i_epoch]
                if best_record is None or (i_auroc + p_auroc > best_record[0] + best_record[3]):
                    best_record = current_eval_values
                    if ckpt_path_best_file and os.path.exists(ckpt_path_best_file):
                        try: os.remove(ckpt_path_best_file)
                        except OSError as e: LOGGER.warning(f"Could not remove old best ckpt {ckpt_path_best_file}: {e}")
                    ckpt_path_best_file = os.path.join(self.ckpt_dir, "ckpt_best_{}.pth".format(i_epoch))
                    torch.save(state_dict, ckpt_path_best_file)
                    LOGGER.info(f"Saved new best checkpoint: {ckpt_path_best_file} (Epoch {i_epoch})")

                pbar_str_val_metrics = (f" IAUC:{current_eval_values[0]*100:.1f}(B:{best_record[0]*100:.1f})"
                                     f" IF1:{current_eval_values[2]*100:.1f}(B:{best_record[2]*100:.1f})"
                                     f" PAUC:{current_eval_values[3]*100:.1f}(B:{best_record[3]*100:.1f})"
                                     f" PF1:{current_eval_values[5]*100:.1f}(B:{best_record[5]*100:.1f})"
                                     f" E:{i_epoch}(BE:{best_record[8]})")

            pbar_current_desc = pbar_str_train_metrics + pbar_str_val_metrics
            pbar.set_description_str(pbar_current_desc)
            torch.save(state_dict, ckpt_path_save)

        pbar.close()
        if best_record is None:
             LOGGER.warning(f"No evaluation performed or no best record found for {name}. Returning zeros.")
             return [0.0] * num_metrics + [-1]
        return best_record

    def _train_discriminator(self, input_data, cur_epoch, pbar, pbar_str_val_metrics):
        self.forward_modules.eval()
        if hasattr(self, 'pre_projection') and self.pre_proj > 0: self.pre_projection.train()
        self.discriminator.train()

        all_loss, all_pt, all_pf = [], [], []
        mean_rt, mean_rg, mean_rf = 0.0, 0.0, 0.0
        sample_num = 0
        pbar_str_iter_desc = ""

        for i_iter, data_item in enumerate(input_data):
            if self.limit > 0 and sample_num >= self.limit: break

            self.dsc_opt.zero_grad()
            if hasattr(self, 'proj_opt') and self.pre_proj > 0: self.proj_opt.zero_grad()
            if hasattr(self, 'backbone_opt') and self.train_backbone: self.backbone_opt.zero_grad()

            aug_imgs = data_item["aug"].to(torch.float).to(self.device)
            real_imgs = data_item["image"].to(torch.float).to(self.device)

            fake_feats_embed, _ = self._embed(aug_imgs, evaluation=False, provide_patch_shapes=True)
            true_feats_embed, _ = self._embed(real_imgs, evaluation=False, provide_patch_shapes=True)

            fake_feats, true_feats = fake_feats_embed, true_feats_embed
            if hasattr(self, 'pre_projection') and self.pre_proj > 0:
                if fake_feats_embed.numel() > 0:
                    fake_feats_proj = self.pre_projection(fake_feats_embed)
                    fake_feats = fake_feats_proj[0] if isinstance(fake_feats_proj, tuple) and len(fake_feats_proj)==2 and fake_feats_proj[0] is not None else fake_feats_proj
                else: fake_feats = fake_feats_embed 

                if true_feats_embed.numel() > 0:
                    true_feats_proj = self.pre_projection(true_feats_embed)
                    true_feats = true_feats_proj[0] if isinstance(true_feats_proj, tuple) and len(true_feats_proj)==2 and true_feats_proj[0] is not None else true_feats_proj
                else: true_feats = true_feats_embed


            if true_feats.numel() == 0 :
                LOGGER.warning(f"_train_discriminator: true_feats is empty for iter {i_iter}. Skipping batch.")
                sample_num += real_imgs.shape[0]
                continue

            current_noise = torch.normal(0, self.noise_val, true_feats.shape).to(self.device)
            gaus_feats = true_feats + current_noise


            discriminator_input = torch.cat([true_feats, gaus_feats], dim=0)

            scores_disc = self.discriminator(discriminator_input)
            s_true = scores_disc[:len(true_feats)]
            s_gaus = scores_disc[len(true_feats):]

            loss_true_bce = F.binary_cross_entropy(s_true, torch.zeros_like(s_true)) 
            loss_gaus_bce = F.binary_cross_entropy(s_gaus, torch.ones_like(s_gaus)) 
            loss = loss_true_bce + loss_gaus_bce

            loss.backward()
            if hasattr(self, 'proj_opt') and self.pre_proj > 0: self.proj_opt.step()
            if hasattr(self, 'backbone_opt') and self.train_backbone: self.backbone_opt.step()
            self.dsc_opt.step()

            with torch.no_grad():
                p_true_val = (s_true.detach() < self.dsc_margin).float().mean() if s_true.numel() > 0 else torch.tensor(0.0)
                p_fake_val = (s_gaus.detach() >= self.dsc_margin).float().mean() if s_gaus.numel() > 0 else torch.tensor(0.0)

            dataset_name_for_log = input_data.name if hasattr(input_data, 'name') else (input_data.dataset.name if hasattr(input_data, 'dataset') and hasattr(input_data.dataset, 'name') else "train_iter")
            self.logger.logger.add_scalar(f"{dataset_name_for_log}/loss", loss.item(), self.logger.g_iter)
            self.logger.logger.add_scalar(f"{dataset_name_for_log}/p_true", p_true_val.item(), self.logger.g_iter)
            self.logger.logger.add_scalar(f"{dataset_name_for_log}/p_fake", p_fake_val.item(), self.logger.g_iter)
            self.logger.step()

            all_loss.append(loss.item())
            all_pt.append(p_true_val.item())
            all_pf.append(p_fake_val.item())
            sample_num += real_imgs.shape[0]

            mean_loss = np.mean(all_loss) if all_loss else 0; mean_pt = np.mean(all_pt) if all_pt else 0; mean_pf = np.mean(all_pf) if all_pf else 0
            pbar_str_iter_desc = f"epoch:{cur_epoch} loss:{mean_loss:.2e} pt:{mean_pt*100:.1f} pf:{mean_pf*100:.1f} svd:{self.svd} sample:{sample_num}/{self.limit}"
            pbar_current_desc = pbar_str_iter_desc + pbar_str_val_metrics
            pbar.set_description_str(pbar_current_desc)

        return pbar_str_iter_desc, (np.mean(all_pt) if all_pt else 0), (np.mean(all_pf) if all_pf else 0)

    def tester(self, test_data, name):
        ckpt_path_list = glob.glob(os.path.join(self.ckpt_dir, 'ckpt_best*'))
        epoch_tested = -1
        default_metrics_values = [0.0] * 8

        if len(ckpt_path_list) > 0:
            best_ckpt_file = ckpt_path_list[0]
            try:
                state_dict_test = torch.load(best_ckpt_file, map_location=self.device)
                if 'discriminator' in state_dict_test:
                    self.discriminator.load_state_dict(state_dict_test['discriminator'])
                    if "pre_projection" in state_dict_test and hasattr(self, 'pre_projection') and self.pre_proj > 0:
                        self.pre_projection.load_state_dict(state_dict_test["pre_projection"])
                elif 'model_state_dict' in state_dict_test:
                    self.load_state_dict(state_dict_test['model_state_dict'], strict=False)
                else: self.load_state_dict(state_dict_test, strict=False)

                images_test, scores_test, segmentations_test, labels_gt_test, masks_gt_test = self.predict(test_data)
                eval_results = self._evaluate(images_test, scores_test, segmentations_test, labels_gt_test, masks_gt_test, name, path='eval')
                i_auroc, i_ap, i_f1, p_auroc, p_ap, p_f1_m, p_iou_m, p_pro = eval_results
                epoch_tested = int(os.path.basename(best_ckpt_file).split('_')[-1].split('.')[0])
                return i_auroc, i_ap, i_f1, p_auroc, p_ap, p_f1_m, p_iou_m, p_pro, epoch_tested
            except Exception as e: LOGGER.error(f"Error loading checkpoint or testing for {name}: {e}"); return default_metrics_values + [epoch_tested]
        else: LOGGER.info(f"No 'ckpt_best*' file found for {name} in {self.ckpt_dir} for testing!"); return default_metrics_values + [epoch_tested]

    def _evaluate(self, images, scores, segmentations, labels_gt, masks_gt, name, path='training'):

        scores_np = np.squeeze(np.array(scores, dtype=np.float32))
        labels_gt_np = np.array(labels_gt, dtype=np.int32)
        image_auroc, image_ap, image_f1 = 0.0, 0.0, 0.0
        if len(scores_np) > 0 and len(labels_gt_np) > 0 and len(np.unique(labels_gt_np)) > 1 :
            try:
                image_metrics_dict = metrics.compute_imagewise_retrieval_metrics(scores_np, labels_gt_np, path=path)
                image_auroc = image_metrics_dict.get("auroc", 0.0); image_ap = image_metrics_dict.get("ap", 0.0); image_f1 = image_metrics_dict.get("max_f1", 0.0)
            except Exception as e: LOGGER.error(f"Error computing image-wise metrics for {name} ({path}): {e}")
        elif len(scores_np) > 0 and len(labels_gt_np) > 0: LOGGER.warning(f"Only one class present in labels_gt for {name} ({path}). Image metrics might be 0 or misleading.")

        pixel_auroc, pixel_ap, pixel_f1_m, pixel_iou_m, pixel_pro = -1.0, -1.0, -1.0, -1.0, -1.0
        if masks_gt is not None and len(masks_gt) > 0:
            segmentations_np = np.array(segmentations, dtype=np.float32)
            masks_gt_np = (np.array(masks_gt, dtype=np.float32) > 0.5).astype(np.uint8)
            masks_gt_to_metrics = masks_gt_np.squeeze(1) if masks_gt_np.ndim == 4 and masks_gt_np.shape[1] == 1 else masks_gt_np
            segmentations_to_metrics = segmentations_np.squeeze(1) if segmentations_np.ndim == 4 and segmentations_np.shape[1] == 1 else segmentations_np
            if segmentations_to_metrics.size>0 and masks_gt_to_metrics.size>0 and segmentations_to_metrics.shape==masks_gt_to_metrics.shape and segmentations_to_metrics.ndim==3 :
                try:
                    pixel_metrics_dict = metrics.compute_pixelwise_retrieval_metrics(segmentations_to_metrics, masks_gt_to_metrics, path=path, segmentation_threshold=self.pixel_metrics_threshold)
                    pixel_auroc = pixel_metrics_dict.get("auroc",0.0); pixel_ap = pixel_metrics_dict.get("ap",0.0); pixel_f1_m = pixel_metrics_dict.get("f1",0.0); pixel_iou_m = pixel_metrics_dict.get("iou",0.0)
                except Exception as e: LOGGER.error(f"Error computing pixel-wise retrieval metrics for {name} ({path}): {e}")
                if path == 'eval':
                    try: pixel_pro = metrics.compute_pro(masks_gt_to_metrics, segmentations_to_metrics)
                    except Exception as e: LOGGER.error(f"Error calculating pixel PRO for {name} ({path}): {e}"); pixel_pro = 0.0
            else: LOGGER.warning(f"Pixel metrics skipped for {name} ({path}) due to empty or mismatched final shapes. Seg shape: {segmentations_to_metrics.shape}, GT shape: {masks_gt_to_metrics.shape}")

        if path != 'training' and len(images) > 0 and ( (len(segmentations) > 0 and len(masks_gt) > 0) or (len(segmentations) == 0 and len(masks_gt) == 0) ) : # Görselleştirme için en azından resimler olmalı
            vis_path_root = os.path.join(self.ckpt_dir, 'visualizations', f"{path}_epoch{self.current_epoch_for_vis}")
            os.makedirs(vis_path_root, exist_ok=True)
            num_to_save = min(len(images), 10)
            for i_vis in range(num_to_save):
                try:
                    img_disp = utils.torch_format_2_numpy_img(images[i_vis])
                    gt_disp = cv2.cvtColor((masks_gt_np[i_vis].squeeze()*255).astype(np.uint8), cv2.COLOR_GRAY2BGR) if masks_gt and i_vis < len(masks_gt_np) else np.zeros_like(img_disp)
                    seg_disp = cv2.applyColorMap((segmentations_np[i_vis].squeeze()*255).astype(np.uint8), cv2.COLORMAP_JET) if segmentations and i_vis < len(segmentations_np) else np.zeros_like(img_disp)
                    combined_vis = np.hstack([img_disp, gt_disp, seg_disp])
                    cv2.imwrite(os.path.join(vis_path_root, f"{name}_{str(i_vis + 1).zfill(3)}.png"), combined_vis)
                except Exception as e: LOGGER.error(f"Error saving visualization for {name} img {i_vis} in {path}: {e}")
        return image_auroc, image_ap, image_f1, pixel_auroc, pixel_ap, pixel_f1_m, pixel_iou_m, pixel_pro

    def _predict(self, img_batch_torch):
        img_batch_torch = img_batch_torch.to(torch.float).to(self.device)
        self.forward_modules.eval()
        if hasattr(self, 'pre_projection') and self.pre_proj > 0: self.pre_projection.eval()
        self.discriminator.eval()

        with torch.no_grad():
            patch_features, patch_shapes = self._embed(img_batch_torch, provide_patch_shapes=True, evaluation=True)
            if patch_features.numel() == 0: 
                LOGGER.warning("_predict: _embed returned empty features. Returning zeros.")
                dummy_scores = np.zeros(img_batch_torch.shape[0])
                h_dummy, w_dummy = self.input_shape[-2:] if self.input_shape and len(self.input_shape)==3 else (256,256)
                dummy_segmentations = [np.zeros((h_dummy,w_dummy)) for _ in range(img_batch_torch.shape[0])]
                return list(dummy_scores), dummy_segmentations

            patch_features_final = patch_features
            if hasattr(self, 'pre_projection') and self.pre_proj > 0:
                patch_features_proj = self.pre_projection(patch_features)
                patch_features_final = patch_features_proj[0] if isinstance(patch_features_proj, tuple) and len(patch_features_proj) == 2 and patch_features_proj[0] is not None else patch_features_proj

            patch_scores_from_discriminator = self.discriminator(patch_features_final)
            unpatched_for_segmentation = self.patch_maker.unpatch_scores(patch_scores_from_discriminator.clone(), batchsize=img_batch_torch.shape[0])

            patch_scores_heatmap_input = unpatched_for_segmentation
            if patch_shapes and patch_shapes[0] and isinstance(patch_shapes[0], (tuple, list)) and len(patch_shapes[0]) == 2:
                h_patch, w_patch = patch_shapes[0]
                try:
                    patch_scores_heatmap_input = unpatched_for_segmentation.reshape(
                        img_batch_torch.shape[0],
                        h_patch,
                        w_patch
                    )
                except Exception as e:
                    LOGGER.warning(
                        f"_predict: Dynamic reshape failed ({e}); "
                        f"falling back to original shape {unpatched_for_segmentation.shape}"
                    )

            segmentation_heatmaps_output = self.anomaly_segmentor.convert_to_segmentation(patch_scores_heatmap_input)

            unpatched_for_scoring = self.patch_maker.unpatch_scores(patch_scores_from_discriminator.clone(), batchsize=img_batch_torch.shape[0])
            image_level_scores_np = self.patch_maker.score(unpatched_for_scoring)
            if isinstance(image_level_scores_np, torch.Tensor): image_level_scores_np = image_level_scores_np.cpu().numpy()

        final_segmentation_list = []
        if isinstance(segmentation_heatmaps_output, np.ndarray) and segmentation_heatmaps_output.ndim >= 2 :
            if segmentation_heatmaps_output.ndim == 3: [final_segmentation_list.append(segmentation_heatmaps_output[i]) for i in range(segmentation_heatmaps_output.shape[0])]
            elif segmentation_heatmaps_output.ndim == 4 and segmentation_heatmaps_output.shape[1] == 1: [final_segmentation_list.append(segmentation_heatmaps_output[i].squeeze(0)) for i in range(segmentation_heatmaps_output.shape[0])]
            elif segmentation_heatmaps_output.ndim == 2 and img_batch_torch.shape[0] == 1 : final_segmentation_list.append(segmentation_heatmaps_output)
            else:
                h_def,w_def=(256,256); LOGGER.error(f"Unexpected segm. shape {segmentation_heatmaps_output.shape}"); [final_segmentation_list.append(np.zeros((h_def,w_def))) for _ in range(img_batch_torch.shape[0])]
        elif isinstance(segmentation_heatmaps_output, list): final_segmentation_list = segmentation_heatmaps_output
        else:
            h_def,w_def=(256,256); LOGGER.error(f"Unexpected segm. type {type(segmentation_heatmaps_output)}"); [final_segmentation_list.append(np.zeros((h_def,w_def))) for _ in range(img_batch_torch.shape[0])]
        return list(image_level_scores_np), final_segmentation_list

    def predict(self, test_dataloader):
        self.forward_modules.eval()
        if hasattr(self, 'pre_projection') and self.pre_proj > 0: self.pre_projection.eval()
        self.discriminator.eval()
        img_paths_all, images_all, scores_all, masks_pred_all, labels_gt_all, masks_gt_all = [], [], [], [], [], []
        with tqdm.tqdm(test_dataloader, desc="Inferring...", leave=False, unit='batch') as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt_all.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask", None) is not None : masks_gt_all.extend(data["mask"].cpu().numpy().tolist())
                    elif data.get("mask_gt", None) is not None : masks_gt_all.extend(data["mask_gt"].cpu().numpy().tolist())
                    image_batch_tensor = data["image"]
                    image_batch_np = image_batch_tensor.cpu().numpy()
                    for j in range(image_batch_np.shape[0]): images_all.append(image_batch_np[j])
                    if "image_path" in data: img_paths_all.extend(data["image_path"]) 
                else:
                    image_batch_tensor = data.to(torch.float).to(self.device) 
                    image_batch_np = image_batch_tensor.cpu().numpy()
                    for j in range(image_batch_np.shape[0]): images_all.append(image_batch_np[j])


                with torch.no_grad(): _scores_batch, _masks_batch = self._predict(image_batch_tensor)
                scores_all.extend(_scores_batch); masks_pred_all.extend(_masks_batch)
        return images_all, scores_all, masks_pred_all, labels_gt_all, masks_gt_all