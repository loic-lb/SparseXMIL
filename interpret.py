import os
import argparse
import cv2
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataset import DatasetPretrained
from models.model_attention import GatedAttention
from models.model_transmil import TransMIL
from models.model_dgcn import DGCNMIL
from models.model_xmil import XMIL
from models.model_utils import ModelEmaV2
from utils import apply_random_seed, get_dataloader, GradCAM, create_heatmap_from_cam, create_heatmap_from_attention,\
    create_mask_from_contours, create_rgb_heatmap
from tile_extraction.WholeSlideImage import WholeSlideImage


def main():
    parser = argparse.ArgumentParser(description="Interpretation of MIL methods")

    parser.add_argument('--experiment_folder', type=str, default='./experiments', metavar='PATH',
                        help='path of folder containing experiments')
    parser.add_argument('--experiment_name', type=str, default='XMIL', metavar='NAME',
                        help='name of the experiment')
    parser.add_argument('--slide_folder', type=str, default='sample_data', metavar='PATH',
                        help='path of folder containing raw slides data')
    parser.add_argument('--feats_folder', type=str, default='sample_data', metavar='PATH',
                        help='path of folder containing feats from patches')
    parser.add_argument('--patches_folder', type=str, default='./patches', metavar='PATH',
                        help='path of folder containing patches coordinates after preprocessing')
    parser.add_argument('--slides_label_filepath', type=str, default='sample_data/labels.csv', metavar='PATH',
                        help='path of CSV-file containing slide labels')

    # Evaluation parameters
    parser.add_argument('--test_time_augmentation', action='store_true', default=False,
                        help='use test time augmentation')
    parser.add_argument('--test_time_augmentation_times', type=int, default=10, metavar='N',
                        help='number of test time augmentation times')
    parser.add_argument('--use_ema', action="store_true", help='exponential moving average used during training')

    # Model parameters
    parser.add_argument('--model', type=str, choices=["attention", "xmil",
                                                      "transmil", "average", "sparseconvmil",
                                                      "dgcn"],
                        default='xception', metavar='MODEL', help='model name')
    parser.add_argument('--transmil_size', type=int, default=512, metavar='SIZE', help='size of the transmil layers')
    parser.add_argument('--sparse-map-downsample', type=int, default=256, help='downsampling factor of the sparse map')

    # Dataset parameters
    parser.add_argument('--n-tiles-per-wsi', type=int, default=0, metavar='SIZE',
                        help='number of tiles to be sampled per WSI')
    parser.add_argument('--perc_tiles_per_wsi', type=float, default=0.2, metavar='SIZE',
                        help='percentage of tiles to be sampled per WSI')
    parser.add_argument('--tile_size', type=int, default=256, metavar='TILESIZE',
                        help='tile size')

    # Interpretation parameters
    parser.add_argument('--sparse_model_layer', type=str, default="middle_flow", choices=["exit_flow", "middle_flow"],
                        help='layer to use for sparse model activations visualization')
    parser.add_argument('--annotated_data', type=str, default=None, help='compute interpretation on annotated data')
    parser.add_argument('--split_id', type=int, default=None, metavar='SPLITID',
                        help='using a specific split id to use '
                             'for interpretation')
    parser.add_argument('--downsample_factor', type=int, default=32, metavar='SIZE', help='downsample factor '
                                                                                          'for heatmap')
    parser.add_argument('--save_heatmaps', action='store_true', default=False, help='save heatmaps')
    parser.add_argument('--save_mask_tissues', action='store_true', default=False, help='save mask of tissue')

    # Miscellaneous parameters
    parser.add_argument('--j', type=int, default=10, metavar='N_WORKERS', help='number of workers for dataloader')
    parser.add_argument('--seed', type=int, default=512, metavar='SEED', help='seed for reproducible experiments')

    args = parser.parse_args()

    apply_random_seed(args.seed)

    print('Loading data')

    test_dataset = DatasetPretrained(args.patches_folder, args.feats_folder, args.slides_label_filepath,
                                     args.n_tiles_per_wsi, percentage=args.perc_tiles_per_wsi,
                                     tile_size=(args.tile_size, args.tile_size), n_workers=args.j)
    n_classes = test_dataset.n_classes
    split_ids = [args.split_id] if args.split_id else range(10)
    for split_id in split_ids:

        output_folder = os.path.join(args.experiment_folder, args.experiment_name, "interpretation_bis",
                                     f"Split {split_id}")
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        apply_random_seed(args.seed)

        print(f"Processing split {split_id}")

        if args.annotated_data:
            annotated_data = pd.read_csv(args.annotated_data)
            test_indices = np.nonzero(np.in1d(test_dataset.slides_ids, annotated_data.slide_id.values))[0]
        else:
            split_csv = pd.read_csv(os.path.join(args.experiment_folder, "splits", f"split_{split_id}.csv"))
            test_indices = np.nonzero(np.in1d(test_dataset.slides_ids, split_csv.test.values))[0]
        test_dataset_split = torch.utils.data.Subset(test_dataset, test_indices)
        # Check that every slide folder has a corresponding label
        assert all([slide_id in test_dataset_split.dataset.slides_labels.keys() for slide_id in
                    test_dataset_split.dataset.slides_ids[test_dataset_split.indices]])
        print(len(test_dataset_split))

        test_dataloader = get_dataloader(test_dataset_split, 1, False, args.j, seed=args.seed)
        print('done')

        # Loads MIL model, optimizer and loss function
        print('Loading model')

        if args.model == 'attention':
            model = GatedAttention(1024, n_classes).cuda()
        elif args.model == 'transmil':
            model = TransMIL(n_classes, args.transmil_size).cuda()
        elif args.model == 'dgcn':
            model = DGCNMIL(num_features=1024, n_classes=n_classes).cuda()
        elif args.model == 'xmil':
            model = XMIL(nb_layers_in=1024, sparse_map_downsample=args.sparse_map_downsample,
                         num_classes=n_classes, perf_aug=False).cuda()
        else:
            raise NotImplementedError
        if args.use_ema:
            model = ModelEmaV2(model, args.model, perf_aug=False, device="cuda")

        experiment_path = os.path.join(args.experiment_folder, args.experiment_name, f"Split {split_id}")
        best_model = torch.load(os.path.join(experiment_path, "best_model.pt"))
        if best_model["settings"] is not None:
            print("Setting model settings to None")
            best_model["settings"] = None
            torch.save(best_model, os.path.join(experiment_path, "best_model.pt"))
        model.load_state_dict(best_model['model'])

        print('Starting inference...')
        model.eval()

        if args.use_ema:
            model = model.module

        if model.name.startswith('Sparse'):
            if args.sparse_model_layer == "exit_flow":
                gradcam = GradCAM(model, model.sparse_model.exit_flow, ["6"])
            elif args.sparse_model_layer == "middle_flow":
                gradcam = GradCAM(model, model.sparse_model.middle_flow, ["7"])
            else:
                raise NotImplementedError

        nb_repeat = args.test_time_augmentation_times if args.test_time_augmentation else 1
        print(nb_repeat)

        grayscale_heatmaps = [[] for _ in range(len(test_dataset))]
        slides_ids = []
        locations_list = [[] for _ in range(len(test_dataset))]
        for i in range(nb_repeat):
            for k, (data, locations, slide_labels, slide_id) in tqdm(enumerate(test_dataloader)):
                locations_list[k].append(locations[0])
                if i == 0:
                    slides_ids.append(slide_id[0])
                if model.name.startswith('Sparse'):
                    grayscale_heatmaps[k].append(create_heatmap_from_cam(gradcam, data[0], locations,
                                                                         args.sparse_map_downsample).T)
                else:
                    if model.name.startswith('DGCN'):
                        with torch.no_grad():
                            _, attention_weights = model(data, locations, return_attention=True)
                    else:
                        with torch.no_grad():
                            try:
                                _, attention_weights = model(data, return_attention=True)
                            except RuntimeError as e:
                                if 'out of memory' in str(e):
                                    print('| WARNING: ran out of memory, skipping batch ...')
                                    torch.cuda.empty_cache()
                                    continue
                    grayscale_heatmaps[k].append(create_heatmap_from_attention(attention_weights, locations[0],
                                                                               args.sparse_map_downsample,
                                                                               args.tile_size,
                                                                               transmil=args.model == 'transmil').T)

        for k in range(len(test_dataset_split)):
            if len(grayscale_heatmaps[k]) == 0:
                print(f"No heatmap found for slide {slides_ids[k]}")

            WSI_object = WholeSlideImage(os.path.join(args.slide_folder, slides_ids[k] + '.svs'), ".svs")
            slide_level = WSI_object.get_best_level_downsample(args.downsample_factor)
            r, b = torch.cat(locations_list[k]).numpy().max(axis=0)
            slide = WSI_object.wsi.read_region((0, 0), slide_level, WSI_object.level_dim[slide_level]).convert('RGB')
            if args.downsample_factor > WSI_object.level_downsamples[slide_level]:
                downsample_gap = WSI_object.level_downsamples[slide_level] / args.downsample_factor
                slide = slide.resize((int(slide.size[0] * downsample_gap), int(slide.size[1] * downsample_gap)))
            r_down, b_down = np.ceil(np.array([r, b]) / args.downsample_factor).astype(int)
            slide = np.array(slide.crop((0, 0, r_down, b_down)))[::, ::, ::-1]
            slide_shape = slide.shape[:2][::-1]
            heatmap_shapes = [heatmap.shape for heatmap in grayscale_heatmaps[k]]
            max_shape = np.max(heatmap_shapes, 0)
            pad_grayscale_heatmaps = [np.pad(heatmap, ((0, max_shape[0] - heatmap.shape[0]),
                                                       (0, max_shape[1] - heatmap.shape[1])), 'constant')
                                      for heatmap in grayscale_heatmaps[k]]
            if "xmil" not in args.model and "sparseconvmil" not in args.model:
                grayscale_heatmap = np.mean(pad_grayscale_heatmaps, 0)
                grayscale_heatmap = cv2.resize(grayscale_heatmap, slide_shape)
            else:
                grayscale_heatmap = np.mean(pad_grayscale_heatmaps, 0)
                grayscale_heatmap = cv2.resize(grayscale_heatmap, slide_shape)
            grayscale_heatmap = (grayscale_heatmap - grayscale_heatmap.min()) / (grayscale_heatmap.max() -
                                                                                 grayscale_heatmap.min())

            if args.save_heatmaps:
                np.save(os.path.join(output_folder, f"{slides_ids[k]}_heatmap.npy"), grayscale_heatmap)
            if "xmil" not in args.model and "sparseconvmil" not in args.model:
                heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_heatmap), cv2.COLORMAP_JET)
            else:
                heatmap = create_rgb_heatmap(grayscale_heatmap)
            mask_tissue = create_mask_from_contours(WSI_object, heatmap.shape[:2], args.downsample_factor)
            if args.save_mask_tissues:
                np.save(os.path.join(output_folder, f"{slides_ids[k]}_mask.npy"), mask_tissue)
            heatmap *= mask_tissue
            superposed_img = cv2.addWeighted(heatmap, 0.5, slide, 0.5, 0)
            cv2.imwrite(os.path.join(output_folder, f"{slides_ids[k]}.png"), superposed_img)


if __name__ == '__main__':
    main()
