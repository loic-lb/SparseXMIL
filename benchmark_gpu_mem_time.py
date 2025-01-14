import argparse
import numpy as np
import time
import os
import pandas as pd
import torch
from tqdm import tqdm
from dataset import DatasetPretrained
from models.model_attention import GatedAttention
from models.model_transmil import TransMIL
from models.model_average import AverageMIL
from models.model_dgcn import DGCNMIL
from models.model_sparseconvmil import SparseConvMIL
from models.model_xmil import XMIL
from models.model_xmil_dense import DenseXMIL
from models.model_nic import NIC
from utils import get_dataloader, apply_random_seed


def perform_epoch(model, dataloader, optimizer, criterion, train=False):
    start_time = time.time()
    for data, locations, slides_labels, slide_id in dataloader:
        if train:
            slides_labels = slides_labels.cuda()
            optimizer.zero_grad()
            if model.name.startswith('Sparse') or model.name.startswith("Dense") or model.name.startswith("NIC"):
                predictions, _ = model(data, locations)
            elif model.name.startswith('DGCN'):
                predictions = model(data, locations)
            else:
                predictions = model(data)
            loss = criterion(predictions, slides_labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                if model.name.startswith('Sparse') or model.name.startswith("Dense") or model.name.startswith("NIC") \
                        or model.name.startswith('DGCN'):
                    model(data, locations)
                else:
                    model(data)
    elapsed_time = time.time() - start_time
    return elapsed_time


def main():
    parser = argparse.ArgumentParser(description='SparseConvMIL: Sparse Convolutional Context-Aware Multiple Instance '
                                                 'Learning for Whole Slide Image Classification')

    parser.add_argument("--experiment_path", type=str,
                        help="path to experiment")
    parser.add_argument('--patches_folder', type=str, default='./patches', metavar='PATH',
                        help='path to folder containing patches coordinates after preprocessing')
    parser.add_argument('--feats_folder', type=str, default='./feats/pt_files', metavar='PATH',
                        help='path to folder containing features extracted from patches')
    parser.add_argument('--slides_label_filepath', type=str, default='./dataset.csv', metavar='PATH',
                        help='path to CSV-file containing slide labels')

    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=16, metavar='SIZE',
                        help='number of slides sampled per iteration')
    parser.add_argument('--training', action="store_true",
                        help="Benchmark model during training (else benchmark during testing)")

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

    # Miscellaneous parameters
    parser.add_argument('--j', type=int, default=10, metavar='N_WORKERS', help='number of workers for dataloader')
    parser.add_argument('--seed', type=int, default=512, metavar='SEED', help='seed for reproducible experiments')

    args = parser.parse_args()

    results_dict = {'Model': args.model}

    # Loads test dataset
    print('Loading data')
    test_dataset = DatasetPretrained(args.patches_folder, args.feats_folder, args.slides_label_filepath,
                                     args.n_tiles_per_wsi, percentage=args.perc_tiles_per_wsi,
                                     tile_size=(args.tile_size, args.tile_size), n_workers=args.j,
                                     sort_tiles=args.model == "transmil")

    print(test_dataset.correspondence_digit_label_name)
    n_classes = test_dataset.n_classes

    split_id = 0
    print(f"Processing split {split_id}")

    split_csv = pd.read_csv(os.path.join("./BRCA", f"splits_{split_id}.csv"))
    if args.training:
        indices = np.nonzero(np.in1d(test_dataset.slides_ids, split_csv.train.values))[0]
    else:
        indices = np.nonzero(np.in1d(test_dataset.slides_ids, split_csv.test.values))[0]
    dataset_split = torch.utils.data.Subset(test_dataset, indices)
    print(len(dataset_split))

    # Check that every slide folder has a corresponding label
    assert all([slide_id in dataset_split.dataset.slides_labels.keys() for slide_id in
                dataset_split.dataset.slides_ids[dataset_split.indices]])

    apply_random_seed(args.seed)

    dataloader = get_dataloader(dataset_split, args.batch_size, args.training, args.j, seed=args.seed)
    print('done')

    # Loads MIL model, optimizer and loss function
    print('Loading model')

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
                              perf_aug=args.training, num_classes=n_classes).cuda()
    elif args.model == "dense_xmil":
        model = DenseXMIL(1024, sparse_map_downsample=args.sparse_map_downsample,
                          num_classes=n_classes, perf_aug=args.training).cuda()
    elif args.model == "nic":
        model = NIC(1024, sparse_map_downsample=args.sparse_map_downsample,
                    num_classes=n_classes, perf_aug=args.training).cuda()
    elif args.model == "xmil":
        model = XMIL(nb_layers_in=1024, sparse_map_downsample=args.sparse_map_downsample,
                     num_classes=n_classes, perf_aug=args.training).cuda()
    else:
        raise NotImplementedError

    print('Starting inference...')

    if not args.training:
        model.eval()

    optimizer = torch.optim.Adam(model.parameters(), 2e-4, weight_decay=1e-7)
    criterion = torch.nn.CrossEntropyLoss()

    epoch_times = []
    for _ in tqdm(range(10)):
        elapsed_time = perform_epoch(model, dataloader, optimizer, criterion, args.training)
        epoch_times.append(elapsed_time)

    print(f"Average elapsed time for model {args.model}: {np.mean(epoch_times)}")
    results_dict["Average elapsed time"] = np.mean(epoch_times)
    result_csv_path = os.path.join(args.experiment_path, f"elapsed_times.csv")
    if os.path.exists(result_csv_path):
        results = pd.read_csv(result_csv_path)
    else:
        results = pd.DataFrame(columns=['Model', 'Average elapsed time'])
    results = pd.concat([results, pd.DataFrame(results_dict, index=[0])], ignore_index=True)
    results.to_csv(result_csv_path, index=False)


if __name__ == '__main__':
    main()
