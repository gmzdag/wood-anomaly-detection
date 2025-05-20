from datetime import datetime
import pandas as pd
import os
import logging
import sys
import click
import torch
import warnings
import backbones
import glass 
import utils
import shutil 

LOGGER = logging.getLogger(__name__)

@click.group(chain=True)
@click.option("--results_path", type=str, default="results", show_default=True)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group", show_default=True)
@click.option("--log_project", type=str, default="project", show_default=True)
@click.option("--run_name", type=str, default="test", show_default=True)
@click.option("--test", type=str, default="ckpt", show_default=True)
def main(**kwargs):
    """Ana CLI grubu."""
    pass


@main.command("net")
@click.option("--dsc_margin", type=float, default=0.5, show_default=True)
@click.option("--train_backbone", is_flag=True, show_default=True)
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
@click.option("--pretrain_embed_dimension", type=int, default=1024, show_default=True)
@click.option("--target_embed_dimension", type=int, default=1024, show_default=True)
@click.option("--patchsize", type=int, default=3, show_default=True)
@click.option("--meta_epochs", type=int, default=640, show_default=True)
@click.option("--eval_epochs", type=int, default=1, show_default=True)
@click.option("--dsc_layers", type=int, default=2, show_default=True)
@click.option("--dsc_hidden", type=int, default=1024, show_default=True)
@click.option("--pre_proj", type=int, default=1, show_default=True)
@click.option("--mining", type=int, default=1, show_default=True)
@click.option("--noise", type=float, default=0.015, show_default=True)
@click.option("--radius", type=float, default=0.75, show_default=True)
@click.option("--p", type=float, default=0.5, show_default=True)
@click.option("--lr", type=float, default=0.0001, show_default=True)
@click.option("--svd", type=int, default=0, show_default=True)
@click.option("--step", type=int, default=20, show_default=True)
@click.option("--limit", type=int, default=392, show_default=True)
def net(
        backbone_names,
        layers_to_extract_from,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize,
        meta_epochs,
        eval_epochs,
        dsc_layers,
        dsc_hidden,
        dsc_margin,
        train_backbone,
        pre_proj,
        mining,
        noise,
        radius,
        p,
        lr,
        svd,
        step,
        limit,
):
    """Network yapılandırma komutu."""
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = []
        for idx in range(len(backbone_names)):
            layers_to_extract_from_coll.append(layers_to_extract_from)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_glass(input_shape_param, device_param):
        glasses = []
        for backbone_name, layers_to_extract_from_single in zip(backbone_names, layers_to_extract_from_coll):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(backbone_name.split("-")[-1])

            current_backbone = backbones.load(backbone_name)
            current_backbone.name, current_backbone.seed = backbone_name, backbone_seed

            glass_inst = glass.GLASS(device_param) 
            glass_inst.load(
                backbone=current_backbone,
                layers_to_extract_from=layers_to_extract_from_single,
                device=device_param,
                input_shape=input_shape_param, 
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                meta_epochs=meta_epochs,
                eval_epochs=eval_epochs,
                dsc_layers=dsc_layers,
                dsc_hidden=dsc_hidden,
                dsc_margin=dsc_margin,
                train_backbone=train_backbone,
                pre_proj=pre_proj,
                mining=mining,
                noise=noise,
                radius=radius,
                p=p,
                lr=lr,
                svd=svd,
                step=step,
                limit=limit,
            )
            glasses.append(glass_inst.to(device_param)) 
        return glasses

    return "get_glass", get_glass


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.argument("aug_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--batch_size", default=8, type=int, show_default=True)
@click.option("--num_workers", default=16, type=int, show_default=True)
@click.option("--resize", default=288, type=int, show_default=True)
@click.option("--imagesize", default=288, type=int, show_default=True) 
@click.option("--rotate_degrees", default=0, type=int, show_default=True)
@click.option("--translate", default=0, type=float, show_default=True)
@click.option("--scale", default=0.0, type=float, show_default=True)
@click.option("--brightness", default=0.0, type=float, show_default=True)
@click.option("--contrast", default=0.0, type=float, show_default=True)
@click.option("--saturation", default=0.0, type=float, show_default=True)
@click.option("--gray", default=0.0, type=float, show_default=True)
@click.option("--hflip", default=0.0, type=float, show_default=True)
@click.option("--vflip", default=0.0, type=float, show_default=True)
@click.option("--distribution", default=0, type=int, show_default=True)
@click.option("--mean", default=0.5, type=float, show_default=True)
@click.option("--std", default=0.1, type=float, show_default=True)
@click.option("--fg", default=1, type=int, show_default=True)
@click.option("--rand_aug", default=1, type=int, show_default=True)
@click.option("--downsampling", default=8, type=int, show_default=True)
@click.option("--augment", is_flag=True, show_default=True)
def dataset(
        name,
        data_path,
        aug_path,
        subdatasets,
        batch_size,
        resize,
        imagesize, 
        num_workers,
        rotate_degrees,
        translate,
        scale,
        brightness,
        contrast,
        saturation,
        gray,
        hflip,
        vflip,
        distribution,
        mean,
        std,
        fg,
        rand_aug,
        downsampling,
        augment,
):
    """Dataset yapılandırma komutu."""
    _DATASETS = {"mvtec": ["datasets.mvtec", "MVTecDataset"], "visa": ["datasets.visa", "VisADataset"],
                 "mpdd": ["datasets.mvtec", "MVTecDataset"], "wfdd": ["datasets.mvtec", "MVTecDataset"], }
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed_val, test_mode, get_name_dl=name):
        dataloaders_list = []
        for subdataset_name in subdatasets:
 
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                source=data_path, 
                anomaly_source_path=aug_path, 
                classname=subdataset_name,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed_val,
   
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=True,
            )
            test_dataloader.name = get_name_dl + "_" + subdataset_name

            if test_mode == 'ckpt':
                train_dataset_params = {
                    'source': data_path,
                    'anomaly_source_path': aug_path,
                    'classname': subdataset_name,
                    'resize': resize,
                    'imagesize': imagesize,
                    'split': dataset_library.DatasetSplit.TRAIN,
                    'seed': seed_val,
                    'rotate_degrees': rotate_degrees,
                    'translate': translate,
                    'brightness_factor': brightness,
                    'contrast_factor': contrast,
                    'saturation_factor': saturation,
                    'gray_p': gray,
                    'h_flip_p': hflip,
                    'v_flip_p': vflip,
                    'scale': scale,
                    'augment': augment, 
                }

                train_dataset = dataset_library.__dict__[dataset_info[1]](**train_dataset_params)


                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    prefetch_factor=2,
                    pin_memory=True,
                )
                train_dataloader.name = test_dataloader.name
                LOGGER.info(f"Dataset {subdataset_name.upper():^20}: train={len(train_dataset)} test={len(test_dataset)}")
            else:
                train_dataloader = test_dataloader
                LOGGER.info(f"Dataset {subdataset_name.upper():^20}: train=0 (using test data for placeholder or no training) test={len(test_dataset)}")

            dataloader_dict = {
                "training": train_dataloader,
                "testing": test_dataloader,
            }
            dataloaders_list.append(dataloader_dict)

        print("\n")
        return dataloaders_list

    return "get_dataloaders", get_dataloaders


@main.result_callback()
def run(
        methods,
        results_path,
        gpu,
        seed,
        log_project,
        log_group,
        run_name,
        test,
):
    """Ana çalıştırma ve sonuç toplama fonksiyonu."""
    methods_dict = {key: item for (key, item) in methods}

    unique_run_save_path = os.path.join(results_path, log_project, log_group, run_name)

    if test == 'ckpt' and os.path.exists(unique_run_save_path):
        LOGGER.info(f"Overwrite mode: Deleting existing run directory for training: {unique_run_save_path}")
        shutil.rmtree(unique_run_save_path, ignore_errors=True)

    os.makedirs(unique_run_save_path, exist_ok=True)
    LOGGER.info(f"Run results will be saved in: {unique_run_save_path}")

    list_of_dataloaders = methods_dict["get_dataloaders"](seed_val=seed, test_mode=test)
    device = utils.set_torch_device(gpu)

    result_collect = []
    all_distribution_data = []

    for dataloader_count, dataloaders_for_class in enumerate(list_of_dataloaders):
        utils.fix_seeds(seed, device)
        dataset_name_from_loader = dataloaders_for_class["training"].name

        img_size_ds_raw = dataloaders_for_class["training"].dataset.imagesize

        img_size_for_shape_calc = img_size_ds_raw
        if isinstance(img_size_ds_raw, (list, tuple)) and len(img_size_ds_raw) == 2 and img_size_ds_raw[0] < 10 : # Genellikle kanal sayısı 3 veya 1 olur
            img_size_for_shape_calc = img_size_ds_raw[1]

        if isinstance(img_size_for_shape_calc, (list, tuple)) and len(img_size_for_shape_calc) == 2:
            current_input_shape = (3, img_size_for_shape_calc[0], img_size_for_shape_calc[1])
        elif isinstance(img_size_for_shape_calc, int):
            current_input_shape = (3, img_size_for_shape_calc, img_size_for_shape_calc)
        else:
            default_img_size = 256 
            current_input_shape = (3, default_img_size, default_img_size)


        glass_list = methods_dict["get_glass"](input_shape_param=current_input_shape, device_param=device) # input_shape, device -> _param

        LOGGER.info(
            "Selecting dataset [{}] ({}/{}) {}".format(
                dataset_name_from_loader,
                dataloader_count + 1,
                len(list_of_dataloaders),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            )
        )

        models_dir_in_main = os.path.join(unique_run_save_path, "models")

        for i, current_glass_model in enumerate(glass_list):
            if current_glass_model.backbone.seed is not None:
                utils.fix_seeds(current_glass_model.backbone.seed, device)

            specific_model_path = os.path.join(models_dir_in_main, f"backbone_{i}")
            current_glass_model.set_model_dir(specific_model_path, dataset_name_from_loader)

            if test == 'ckpt':
                training_outcome = current_glass_model.trainer(
                    dataloaders_for_class["training"],
                    dataloaders_for_class["testing"],
                    dataset_name_from_loader
                )

                if isinstance(training_outcome, int):
                    svd_value_from_trainer = training_outcome
                    all_distribution_data.append({
                        'Class': dataset_name_from_loader,
                        'Distribution': svd_value_from_trainer,
                        'Foreground': svd_value_from_trainer
                    })
                    continue
                else:
                    if training_outcome and len(training_outcome) == 9:
                        i_auroc, i_ap, i_f1_val, p_auroc, p_ap, p_f1_val, p_iou_val, p_pro, epoch_val = training_outcome

                        result_dict_current = {
                            "dataset_name": dataset_name_from_loader,
                            "image_auroc": i_auroc, "image_ap": i_ap, "image_f1": i_f1_val,
                            "pixel_auroc": p_auroc, "pixel_ap": p_ap, "pixel_f1": p_f1_val,
                            "pixel_iou": p_iou_val, "pixel_pro": p_pro,
                            "best_epoch": epoch_val,
                        }
                        result_collect.append(result_dict_current)
                    else:
                        continue
            else:
                 test_metrics = current_glass_model.tester(
                     dataloaders_for_class["testing"], dataset_name_from_loader
                 )
                 if test_metrics and len(test_metrics) == 9:
                    i_auroc, i_ap, i_f1_test, p_auroc, p_ap, p_f1_test, p_iou_test, p_pro, epoch_test = test_metrics
                    result_dict_current = {
                        "dataset_name": dataset_name_from_loader,
                        "image_auroc": i_auroc, "image_ap": i_ap, "image_f1": i_f1_test,
                        "pixel_auroc": p_auroc, "pixel_ap": p_ap, "pixel_f1": p_f1_test,
                        "pixel_iou": p_iou_test, "pixel_pro": p_pro,
                        "best_epoch": epoch_test,
                    }
                    result_collect.append(result_dict_current)
                 else:
                    continue

            if result_collect and result_collect[-1]["dataset_name"] == dataset_name_from_loader :
                LOGGER.info(f"Results for {dataset_name_from_loader}:")
                current_result_to_print = result_collect[-1]
                if current_result_to_print.get("best_epoch", -1) > -1 :
                    output_str_parts = []
                    for key, item in current_result_to_print.items():
                        if isinstance(item, str):
                            continue
                        elif isinstance(item, int):
                            output_str_parts.append(f"{key}: {item}")
                        else:
                            output_str_parts.append(f"{key}: {item * 100:.2f}")
                    print("  " + " | ".join(output_str_parts))
                    print("-" * 50)

                    if result_collect :
                        csv_metric_names = list(result_collect[0].keys())[1:]
                        csv_dataset_names = [res["dataset_name"] for res in result_collect]
                        csv_scores = [list(res.values())[1:] for res in result_collect]

                        utils.compute_and_store_final_results(
                            unique_run_save_path,
                            csv_scores,
                            csv_metric_names,
                            row_names=csv_dataset_names,
                        )

    if all_distribution_data:
        df_dist = pd.DataFrame(all_distribution_data)
        if not df_dist.empty:
            base_dataset_name_for_excel = "unknown_dataset"
            if 'dataset_name_from_loader' in locals() and dataset_name_from_loader:
                 base_dataset_name_for_excel = dataset_name_from_loader.split('_')[0] if '_' in dataset_name_from_loader else dataset_name_from_loader

            excel_dir = './datasets/excel'
            os.makedirs(excel_dir, exist_ok=True)
            xlsx_path = os.path.join(excel_dir, f'{base_dataset_name_for_excel}_distribution.xlsx')
            try:
                df_dist.to_excel(xlsx_path, index=False)
                LOGGER.info(f"Distribution judgment saved to {xlsx_path}")
            except Exception as e:
                LOGGER.error(f"Could not save distribution judgment to {xlsx_path}: {e}")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()