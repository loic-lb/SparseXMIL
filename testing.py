import argparse
import os
import torch
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.preprocessing import label_binarize
from torch.nn.functional import softmax
from dataset import DatasetPretrained
from models.model_attention import GatedAttention
from models.model_transmil import TransMIL
from models.model_average import AverageMIL
from models.model_dgcn import DGCNMIL
from models.model_sparseconvmil import SparseConvMIL
from models.model_xmil import XMIL
from models.model_xmil_dense import DenseXMIL
from models.model_nic import NIC
from utils import get_dataloader
from models.model_utils import ModelEmaV2
from utils import apply_random_seed


def main():
    parser = argparse.ArgumentParser(description='SparseConvMIL: Sparse Convolutional Context-Aware Multiple Instance '
                                                 'Learning for Whole Slide Image Classification')

    parser.add_argument('--experiment_folder', type=str, default='./experiments', metavar='PATH',
                        help='path to folder containing experiments')
    parser.add_argument('--experiment_name', type=str, default='XMIL', metavar='NAME',
                        help='name of the experiment')
    parser.add_argument('--patches_folder', type=str, default='./patches', metavar='PATH',
                        help='path to folder containing patches coordinates after preprocessing')
    parser.add_argument('--feats_folder', type=str, default='./feats/pt_files', metavar='PATH',
                        help='path to folder containing features extracted from patches')
    parser.add_argument('--slides_label_filepath', type=str, default='./dataset.csv', metavar='PATH',
                        help='path to CSV-file containing slide labels')

    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=16, metavar='SIZE',
                        help='number of slides sampled per iteration')
    parser.add_argument('--test_time_augmentation', action='store_true', default=False,
                        help='perform test time augmentation during testing')
    parser.add_argument('--test_time_augmentation_times', type=int, default=10, metavar='N',
                        help='number of test time augmentation iterations')
    parser.add_argument('--use_ema', action="store_true", help='exponential moving average used during training')
    parser.add_argument('--remove_perf_image_aug', action="store_false", help='remove image augmentation during training')

    # Model parameters
    parser.add_argument('--model', type=str, choices=["attention", "xmil",
                                                      "transmil", "average", "sparseconvmil",
                                                      "dgcn", "dense_xmil", "nic"],
                        default='xception', metavar='MODEL', help='model name')
    parser.add_argument('--transmil_size', type=int, default=512, metavar='SIZE', help='size of the TransMIL layers')
    parser.add_argument('--sparse-map-downsample', type=int, default=256, help='downsampling factor of the sparse map')

    # Dataset parameters
    parser.add_argument('--n-tiles-per-wsi', type=int, default=0, metavar='SIZE',
                        help='number of tiles to be sampled per WSI')
    parser.add_argument('--perc_tiles_per_wsi', type=float, default=0.2, metavar='SIZE',
                        help='percentage of tiles to be sampled per WSI')
    parser.add_argument('--tile_size', type=int, default=256, metavar='TILESIZE',
                        help='tile size')

    # Sensitivity analysis parameters
    parser.add_argument('--shuffle_locations', action='store_true', default=False, help='Shuffle locations')
    parser.add_argument('--shuffle_mode', type=str, default="idx", choices=["idx", "absolute_positions"],
                        help='Shuffle mode')

    # Miscellaneous parameters
    parser.add_argument('--j', type=int, default=10, metavar='N_WORKERS', help='number of workers for dataloader')
    parser.add_argument('--seed', type=int, default=512, metavar='SEED', help='seed for reproducible experiments')

    args = parser.parse_args()

    # Loads test dataset
    print('Loading data')
    test_dataset = DatasetPretrained(args.patches_folder, args.feats_folder, args.slides_label_filepath,
                                     args.n_tiles_per_wsi, percentage=args.perc_tiles_per_wsi,
                                     tile_size=(args.tile_size, args.tile_size), n_workers=args.j,
                                     sort_tiles=args.model == "transmil")

    print(test_dataset.correspondence_digit_label_name)
    n_classes = test_dataset.n_classes

    results_dict = {'Model': args.model}
    # Loops over the splits (assuming 10-fold cross-validation)
    for split_id in range(10):

        apply_random_seed(args.seed)

        print(f"Processing split {split_id}")

        split_csv = pd.read_csv(os.path.join(args.experiment_folder, "splits", f"split_{split_id}.csv"))
        test_indices = np.nonzero(np.in1d(test_dataset.slides_ids, split_csv.test.values))[0]
        test_dataset_split = torch.utils.data.Subset(test_dataset, test_indices)
        print(len(test_dataset_split))

        # Check that every slide folder has a corresponding label
        assert all([slide_id in test_dataset_split.dataset.slides_labels.keys() for slide_id in
                    test_dataset_split.dataset.slides_ids[test_dataset_split.indices]])
        test_dataloader = get_dataloader(test_dataset_split, args.batch_size, False, args.j, seed=args.seed)
        print('done')

        # Loads MIL model, optimizer and loss function
        print('Loading model')
        perf_aug = args.test_time_augmentation and args.remove_perf_image_aug

        if args.model == 'attention':
            model = GatedAttention(1024, n_classes).cuda()
        elif args.model == 'transmil':
            model = TransMIL(n_classes, args.transmil_size).cuda()
        elif args.model == 'average':
            model = AverageMIL(1024, n_classes).cuda()
        elif args.model == 'dgcn':
            model = DGCNMIL(num_features=1024, n_classes=n_classes).cuda()
        elif args.model == 'sparseconvmil':
            model = SparseConvMIL(1024, sparse_map_downsample=args.sparse_map_downsample,
                                  perf_aug=perf_aug, num_classes=n_classes).cuda()
        elif args.model == "dense_xmil":
            model = DenseXMIL(1024, sparse_map_downsample=args.sparse_map_downsample,
                           num_classes=n_classes, perf_aug=perf_aug).cuda()
        elif args.model == "nic":
            model = NIC(1024, sparse_map_downsample=args.sparse_map_downsample,
                           num_classes=n_classes, perf_aug=perf_aug).cuda()
        elif args.model == "xmil":
            model = XMIL(nb_layers_in=1024, sparse_map_downsample=args.sparse_map_downsample,
                         num_classes=n_classes, perf_aug=perf_aug).cuda()
        else:
            raise NotImplementedError

        if args.use_ema:
            model = ModelEmaV2(model, args.model, perf_aug=perf_aug, device="cuda")

        # Retrieves best validation model by looking last epoch saved
        experiment_path = os.path.join(args.experiment_folder, args.experiment_name, f"Split {split_id}")
        if "best_model.pt" in os.listdir(experiment_path):
            best_model = torch.load(os.path.join(experiment_path, "best_model.pt"))
        else:
            checkpoints = sorted([file for file in os.listdir(experiment_path) if file.endswith(".pt")],
                                 key=lambda x: int(x.split('_')[1].split('.')[0]))
            best_model = torch.load(os.path.join(experiment_path, checkpoints[-1]))
            model.load_state_dict(best_model['model'])
            print(f'Best model found: {checkpoints[-1]}')
        if best_model["settings"] is not None:
            print("Setting model settings to None")
            best_model["settings"] = None
            torch.save(best_model, os.path.join(experiment_path, "best_model.pt"))
        model.load_state_dict(best_model['model'])

        print('Starting inference...')
        model.eval()

        if args.use_ema:
            model = model.module

        proba_predictions_final = []

        nb_repeat = args.test_time_augmentation_times if args.test_time_augmentation else 1
        print(nb_repeat)

        for i in range(nb_repeat):
            proba_predictions = []
            ground_truths = []
            for data, locations, slides_labels, slide_id in test_dataloader:

                # Modify data or locations when sensitivity analysis is performed
                if args.shuffle_locations:
                    if model.name.startswith('Trans'):
                        if args.shuffle_mode == "idx":
                            data = [x[torch.randperm(x.shape[0])] for x in data]  # Shuffle the order of the tiles
                            # embeddings for each slide in the batch
                        else:
                            raise NotImplementedError
                    elif model.name.startswith('Sparse') or model.name.startswith('DGCN') \
                            or model.name.startswith("Dense") or model.name.startswith("NIC"):
                        if args.shuffle_mode == "idx":
                            locations = [loc[torch.randperm(loc.shape[0])] for loc in locations]  # Shuffle the
                            # coordinates for each slide in the batch
                        elif args.shuffle_mode == "absolute_positions":
                            locations = [torch.hstack([torch.FloatTensor(loc.shape[0], 1).uniform_(loc[:, 0].min(),
                                                                                                   loc[:, 0].max()),
                                                       torch.FloatTensor(loc.shape[0], 1).uniform_(loc[:, 1].min(),
                                                                                                   loc[:, 1].max())]).int()
                                         for loc in locations]  # Assign random coordinates to each tile within
                            # the maximum and minimum coordinates of each slide in the batch
                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError
                slides_labels = slides_labels.cuda()

                with torch.no_grad():
                    if model.name.startswith('Sparse') or model.name.startswith("Dense") or model.name.startswith("NIC"):
                        predictions, _ = model(data, locations)
                    elif model.name.startswith('DGCN'):
                        predictions = model(data, locations)
                    else:
                        predictions = model(data)

                predictions = softmax(predictions, dim=-1)

                # Store data for final epoch average measures
                proba_predictions.extend(predictions.detach().cpu().numpy())
                ground_truths.extend(slides_labels.cpu().numpy())
            if i == 0:
                ground_truths_old = ground_truths
            else:
                assert ground_truths_old == ground_truths
            proba_predictions_final.append(proba_predictions)
        proba_predictions_final = np.mean(proba_predictions_final, axis=0)

        # Compute the average AUC over all TTA iterations based on the average probabilities of each TTA
        # iteration
        if n_classes == 2:
            auc_score = metrics.roc_auc_score(ground_truths, proba_predictions_final[:, 1])
        else:
            aucs = []
            binary_labels = label_binarize(ground_truths, classes=[i for i in range(n_classes)])
            for class_idx in range(n_classes):
                if class_idx in ground_truths:
                    fpr, tpr, _ = metrics.roc_curve(binary_labels[:, class_idx], proba_predictions_final[:, class_idx])
                    aucs.append(metrics.auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            auc_score = np.nanmean(np.array(aucs))

        results_dict[f"Split {split_id}"] = auc_score

    # Save results to CSV file
    result_csv_path = os.path.join(args.experiment_folder, f"results_TTA_{args.test_time_augmentation}.csv") \
        if not args.shuffle_locations \
        else os.path.join(args.experiment_folder, f"results_TTA_{args.test_time_augmentation}"
                                                  f"_shuffle_{args.shuffle_mode}.csv")
    if os.path.exists(result_csv_path):
        results = pd.read_csv(result_csv_path)
    else:
        results = pd.DataFrame(columns=['Model'] + ['Split ' + str(i) for i in range(10)])
    results = pd.concat([results, pd.DataFrame(results_dict, index=[0])], ignore_index=True)
    results.to_csv(result_csv_path, index=False)


if __name__ == '__main__':
    main()
