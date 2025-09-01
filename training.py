import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from models.model_attention import GatedAttention
from models.model_transmil import TransMIL
from models.model_average import AverageMIL
from models.model_dgcn import DGCNMIL
from torch.nn.functional import softmax
from dataset import DatasetPretrained
from models.model_sparseconvmil import SparseConvMIL
from models.model_xmil import XMIL
from models.model_xmil_dense import DenseXMIL
from models.model_nic import NIC
from utils import get_dataloader, apply_random_seed, split_dataset, measure_perf
from models.model_utils import ModelEmaV2

from pathlib import Path

def _define_args():
    parser = argparse.ArgumentParser(description='SparseConvMIL: Sparse Convolutional Context-Aware Multiple Instance '
                                                 'Learning for Whole Slide Image Classification')

    parser.add_argument('--experiment_folder', type=str, default='./experiments', metavar='PATH',
                        help='path to folder containing experiments')
    parser.add_argument('--experiment_name', type=str, default='XMIL', metavar='NAME',
                        help='name of the experiment')
    parser.add_argument('--split_id', type=int, default=0, metavar='N', help='split id')

    parser.add_argument('--patches_folder', type=str, default='./patches', metavar='PATH',
                        help='path to folder containing patches coordinates after preprocessing')
    parser.add_argument('--feats_folder', type=str, default='./feats/pt_files', metavar='PATH',
                        help='path to folder containing features extracted from patches')
    parser.add_argument('--slides_label_filepath', type=str, default='./dataset.csv', metavar='PATH',
                        help='path to CSV-file containing slide labels')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-3, metavar='LR', help='learning rate')
    parser.add_argument('--reg', type=float, default=1e-6, metavar='R', help='weight decay')
    parser.add_argument('--optimizer', type=str, choices=["Adam", "AdamW"], metavar='OPTIMIZER', default="Adam",
                        help='optimizer to use')
    parser.add_argument('--batch_size', type=int, default=16, metavar='SIZE',
                        help='number of slides sampled per iteration')
    parser.add_argument('--clip', type=float, default=None, metavar='CLIP',
                        help="Perform gradient clipping")

    # Evaluation parameters
    parser.add_argument('--test_time_augmentation', action="store_true", help='perform test time augmentation during'
                                                                              'validation')
    parser.add_argument('--test_time_augmentation_times', type=int, default=10, metavar='N',
                        help='number of test time augmentation iterations')

    # Model parameters
    parser.add_argument('--model', type=str, choices=["attention", "xmil",
                                                      "transmil", "average",
                                                      "sparseconvmil", "dgcn",
                                                      "dense_xmil",
                                                      "nic"],
                        default='xmil',
                        metavar='MODEL',
                        help='model name')
    parser.add_argument('--transmil_size', type=int, default=512, choices=[256, 512],
                        metavar='SIZE', help='size of the TransMIL layers')
    parser.add_argument('--sparse-map-downsample', type=int, default=10, help='downsampling factor of the sparse map')
    parser.add_argument('--remove_perf_image_aug', action="store_false", help='remove image augmentation during training')

    # Dataset parameters
    parser.add_argument('--split', type=str, default=None, help="path to predetermined splitting of dataset")
    parser.add_argument('--n_tiles_per_wsi', type=int, default=0, metavar='SIZE',
                        help='number of tiles to be sampled per WSI')
    parser.add_argument('--perc_tiles_per_wsi', type=float, default=None, metavar='SIZE',
                        help='percentage of tiles to be sampled per WSI')
    parser.add_argument('--tile_size', type=int, default=256, metavar='TILESIZE',
                        help='tile size')

    # Miscellaneous parameters
    parser.add_argument('--j', type=int, default=10, metavar='N_WORKERS', help='number of workers for dataloader')
    parser.add_argument('--seed', type=int, default=512, metavar='SEED', help='seed for reproducible experiments')

    args = parser.parse_args()

    experiment_path = os.path.join(args.experiment_folder, args.experiment_name, f"Split {args.split_id}")
    Path(experiment_path).mkdir(parents=True, exist_ok=True)

    hyper_parameters = {
        'experiment_path': experiment_path,
        'patches_folder': args.patches_folder,
        'feats_folder': args.feats_folder,
        'slides_label_filepath': args.slides_label_filepath,
        'tile_size': (args.tile_size, args.tile_size),
        'epochs': args.epochs,
        'lr': args.lr,
        'reg': args.reg,
        'TTA': args.test_time_augmentation,
        'TTA_times': args.test_time_augmentation_times,
        'model': args.model,
        'transmil_size': args.transmil_size,
        'split': args.split,
        'sparse_map_downsample': args.sparse_map_downsample,
        'perf_image_aug': args.remove_perf_image_aug,
        'batch_size': args.batch_size,
        'clip': args.clip,
        'n_tiles_per_wsi': args.n_tiles_per_wsi,
        'perc_tiles_per_wsi': args.perc_tiles_per_wsi,
        'j': args.j,
        'optimizer': args.optimizer,
        'seed': args.seed,
    }

    return hyper_parameters


def perform_epoch(mil_model, mil_model_ema, dataloader, optimizer, loss_function, clip=None, train=True):
    """
    Perform a complete training/validation epoch by looping through all data of the dataloader.
    :param mil_model: MIL model to be trained
    :param mil_model_ema: EMA version of the MIL model
    :param dataloader: loader of the dataset
    :param optimizer: pytorch optimizer
    :param loss_function: loss function to compute gradients
    :param train: boolean indicating if training or validation is performed (True for training)
    :return: (losses, logits, ground_truths, predicted classes)
    """
    proba_predictions = []
    ground_truths = []
    losses = []

    start_time = time.time()
    for data, locations, slides_labels, slides_ids in dataloader:
        slides_labels = slides_labels.cuda()

        if train:
            optimizer.zero_grad()
            if mil_model.name.startswith('Sparse') or mil_model.name.startswith('Dense') or mil_model.name.startswith(
                    "NIC"):
                predictions, tiles_locations = mil_model(data, locations)
            elif mil_model.name.startswith('DGCN'):
                predictions = mil_model(data, locations)
            else:
                predictions = mil_model(data)
            loss = loss_function(predictions, slides_labels)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(mil_model.parameters(), clip)
            optimizer.step()

            mil_model_ema.update(mil_model)

        else:
            with torch.no_grad():
                if mil_model.name.startswith('Sparse') or mil_model.name.startswith('Dense') \
                        or mil_model.name.startswith("NIC"):
                    predictions, _ = mil_model_ema.module(data, locations)
                elif mil_model.name.startswith('DGCN'):
                    predictions = mil_model_ema.module(data, locations)
                else:
                    predictions = mil_model_ema.module(data)

                loss = loss_function(predictions, slides_labels)

        training_time = time.time() - start_time

        predictions = softmax(predictions, dim=-1)

        # Store data for finale epoch average measures
        losses.append(loss.detach().cpu().numpy())
        proba_predictions.extend(predictions.detach().cpu().numpy())
        ground_truths.extend(slides_labels.cpu().numpy())

    proba_predictions = np.array(proba_predictions)
    predicted_classes = np.argmax(proba_predictions, axis=1)

    return losses, proba_predictions, ground_truths, predicted_classes, training_time


def main(hyper_parameters):
    # Set seed
    apply_random_seed(hyper_parameters['seed'])

    # Loads dataset and dataloader
    print('Loading data')
    dataset = DatasetPretrained(hyper_parameters['patches_folder'], hyper_parameters['feats_folder'],
                                hyper_parameters['slides_label_filepath'], hyper_parameters['n_tiles_per_wsi'],
                                percentage=hyper_parameters['perc_tiles_per_wsi'],
                                tile_size=hyper_parameters['tile_size'],
                                n_workers=hyper_parameters['j'],
                                sort_tiles=hyper_parameters['model'] == "transmil")

    n_classes = dataset.n_classes
    train_dataset, val_dataset = split_dataset(dataset, 0.1, hyper_parameters['split'])

    # Check that every slide has a corresponding label
    assert all([slide_id in train_dataset.dataset.slides_labels.keys() for slide_id in
                train_dataset.dataset.slides_ids[train_dataset.indices]])
    assert all([slide_id in val_dataset.dataset.slides_labels.keys() for slide_id in
                val_dataset.dataset.slides_ids[val_dataset.indices]])

    train_dataloader = get_dataloader(train_dataset, hyper_parameters['batch_size'], True, hyper_parameters['j'],
                                      seed=hyper_parameters['seed'])
    val_dataloader = get_dataloader(val_dataset, hyper_parameters['batch_size'], False, hyper_parameters['j'],
                                    seed=hyper_parameters['seed'])
    print('done')

    # Loads MIL model, optimizer and loss function
    print('Loading model')
    if hyper_parameters['model'] == 'attention':
        model = GatedAttention(1024, n_classes).cuda()
    elif hyper_parameters['model'] == 'transmil':
        model = TransMIL(n_classes, hyper_parameters["transmil_size"]).cuda()
    elif hyper_parameters['model'] == 'average':
        model = AverageMIL(1024, n_classes).cuda()
    elif hyper_parameters['model'] == 'dgcn':
        model = DGCNMIL(num_features=1024, n_classes=n_classes).cuda()
    elif hyper_parameters['model'] == 'sparseconvmil':
        model = SparseConvMIL(1024, sparse_map_downsample=hyper_parameters['sparse_map_downsample'],
                              perf_aug=hyper_parameters['perf_image_aug'], num_classes=n_classes).cuda()
    elif hyper_parameters['model'] == 'dense_xmil':
        model = DenseXMIL(1024, sparse_map_downsample=hyper_parameters['sparse_map_downsample'],
                          num_classes=n_classes, perf_aug=hyper_parameters['perf_image_aug']).cuda()
    elif hyper_parameters['model'] == 'nic':
        model = NIC(1024, sparse_map_downsample=hyper_parameters['sparse_map_downsample'],
                    num_classes=n_classes, perf_aug=hyper_parameters['perf_image_aug']).cuda()
    else:
        model = XMIL(1024, sparse_map_downsample=hyper_parameters['sparse_map_downsample'],
                     num_classes=n_classes, perf_aug=hyper_parameters['perf_image_aug']).cuda()

    # Create EMA version of the model
    model_ema = ModelEmaV2(model, hyper_parameters['model'], perf_aug=hyper_parameters['TTA'] and hyper_parameters['perf_image_aug'], decay=0.99,
                           device="cuda")

    print('  done')

    if hyper_parameters['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), hyper_parameters['lr'],
                                     weight_decay=hyper_parameters['reg'])
    elif hyper_parameters['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), hyper_parameters['lr'],
                                      weight_decay=hyper_parameters['reg'])
    else:
        raise ValueError('Optimizer not supported')
    loss_function = torch.nn.CrossEntropyLoss()

    # Loop through all epochs
    print('Starting training...')
    best_val_auc = 0
    save_every = 10 if hyper_parameters['model'] == 'average' else 1  # Save every 10 epochs for average model
    val_perfs = []

    for epoch in range(hyper_parameters["epochs"]):
        train_losses, train_probas, \
            train_ground_truths, train_predicted_classes, train_time = perform_epoch(model, model_ema,
                                                                                     train_dataloader, optimizer,
                                                                                     loss_function,
                                                                                     clip=hyper_parameters['clip'])

        train_loss, train_bac, train_f1, train_auc = measure_perf(train_losses, train_ground_truths,
                                                                  train_predicted_classes, train_probas,
                                                                  n_classes)
        print('Epoch', f'{epoch:3d}/{hyper_parameters["epochs"]}', f'    train_loss={train_loss:.3f}',
              f'    train_bac={train_bac:.3f}', f'    train_f1={train_f1:.3f}', f'    train_auc={train_auc:.3f}')
        model.eval()

        nb_repeat = hyper_parameters['TTA_times'] if hyper_parameters['TTA'] else 1

        val_loss_avg = []
        val_probas_avg = []

        for sampling_id in range(nb_repeat):

            val_losses, val_probas, \
                val_ground_truths, val_predicted_classes, val_time = perform_epoch(model,
                                                                                   model_ema,
                                                                                   val_dataloader, optimizer,
                                                                                   loss_function, train=False)
            # Keep track of the performance for each TTA iteration
            val_loss, val_bac, val_f1, val_auc = measure_perf(val_losses, val_ground_truths,
                                                              val_predicted_classes, val_probas,
                                                              n_classes)

            # Check that the ground truths are the same for each test time augmentation (TTA) iteration
            if sampling_id == 0:
                val_ground_truths_old = val_ground_truths
            else:
                assert val_ground_truths_old == val_ground_truths

            val_perfs.append(
                {'epoch': epoch, 'sampling_id': sampling_id, 'val_loss': val_loss, 'val_bac': val_bac,
                 'val_f1': val_f1, 'val_auc': val_auc})
            val_loss_avg.append(val_loss)
            val_probas_avg.append(val_probas)

        # Compute the average performance over all TTA iterations based on the average probabilities of each TTA
        # iteration
        val_probas_avg = np.mean(val_probas_avg, axis=0)
        val_predicted_classes_avg = np.argmax(val_probas_avg, axis=1)

        val_loss, val_bac, val_f1, val_auc = measure_perf(val_loss_avg, val_ground_truths,
                                                          val_predicted_classes_avg, val_probas_avg,
                                                          n_classes)

        print('Epoch', f'{epoch:3d}/{hyper_parameters["epochs"]}', f'    val_loss={val_loss:.3f}',
              f'    val_bac {val_bac:.3f}', f'    val_f1={val_f1:.3f}', f'    val_auc={val_auc:.3f}')

        # Save model if best performing model on val dataset
        if (best_val_auc < val_auc) and (epoch % save_every == 0):
            print("New best performing model on val dataset, saving model....")
            checkpoint = {
                'model': model_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'settings': hyper_parameters,
                'epoch': epoch,
            }
            torch.save(checkpoint,
                       f"./{hyper_parameters['experiment_path']}/model_{epoch}.pt")
            best_val_auc = val_auc
            print(f"New best val auc:{best_val_auc}")

    # Save val results to csv file to check difference between TTA iterations
    df = pd.DataFrame(val_perfs)
    df.to_csv(f"./{hyper_parameters['experiment_path']}/val_results.csv", index=False)


print('  done')

if __name__ == '__main__':
    main(_define_args())
