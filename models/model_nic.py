import warnings

import torch
import random
import math
import torch.nn as nn
import torch.nn.functional as F

from MinkowskiEngine import SparseTensor
from MinkowskiEngine import SparseTensorQuantizationMode
from MinkowskiEngine.utils import batched_coordinates

from models.model_xception_dense import SeparableConv2d


def conv_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    return nn.Sequential(
        SeparableConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.2))


class NIC(nn.Module):
    def __init__(self, nb_layers_in, sparse_map_downsample, perf_aug, num_classes,
                 tile_coordinates_rotation_augmentation=True, tile_coordinates_flips_augmentation=True,
                 tile_coordinates_resize_augmentation=True):
        super(NIC, self).__init__()

        self.name = "NIC"
        self.min_size = (65, 65)
        self.nb_layers_in = nb_layers_in
        self.sparse_map_downsample = sparse_map_downsample
        self.perf_aug = perf_aug
        self.num_classes = num_classes
        self.tile_coordinates_rotation_augmentation = tile_coordinates_rotation_augmentation
        self.tile_coordinates_flips_augmentation = tile_coordinates_flips_augmentation
        self.tile_coordinates_resize_augmentation = tile_coordinates_resize_augmentation

        self.adapt_layer = self.get_adapt_layer()

        self.model = self.create_model()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = self.get_classifier()

    def get_adapt_layer(self):
        return nn.Sequential(nn.Linear(1024, 128), nn.ReLU())

    def create_model(self):
        return nn.Sequential(conv_block(128, 128, kernel_size=3, stride=2, padding=1),
                             conv_block(128, 128, kernel_size=3, stride=2, padding=1),
                             conv_block(128, 128, kernel_size=3, stride=2, padding=1),
                             conv_block(128, 128, kernel_size=3, stride=2, padding=1),
                             conv_block(128, 128, kernel_size=3, stride=2, padding=1),
                             conv_block(128, 128, kernel_size=3, stride=2, padding=1),
                             conv_block(128, 128, kernel_size=3, stride=1, padding=1),
                             conv_block(128, 128, kernel_size=3, stride=1, padding=1))

    def get_classifier(self):
        return nn.Sequential(nn.Linear(128, 128),
                             #nn.BatchNorm1d(128),
                             nn.LeakyReLU(),
                             nn.Linear(128, self.num_classes))

    def data_augment_tiles_locations(self, tiles_locations):
        """
        Perform data augmentation of the sparse map of tiles embeddings. First, a matrix of random rotations, flips,
            and resizes is instantiated. Then, a random translation vector is instantiated. The random translation is
            applied on the tiles coordinates, followed by the random rot+flips+resizes.
        :param tiles_locations: matrix of shape (batch_size, n_tiles_per_batch, 2) with tiles coordinates
        :return:
        """
        device = tiles_locations.device

        transform_matrix = torch.eye(2)
        # Random rotations
        if self.tile_coordinates_rotation_augmentation:
            theta = random.uniform(-180., 180.)
            rot_matrix = torch.tensor([[math.cos(theta), -math.sin(theta)],
                                       [math.sin(theta), math.cos(theta)]])
            transform_matrix = rot_matrix
        # Random flips
        if self.tile_coordinates_flips_augmentation:
            flip_h = random.choice([-1., 1.])
            flip_v = random.choice([-1., 1.])
            flip_matrix = torch.tensor([[flip_h, 0.],
                                        [0., flip_v]])
            transform_matrix = torch.mm(transform_matrix, flip_matrix)
        # Random resizes per axis
        if self.tile_coordinates_resize_augmentation:
            size_factor_h = 0.6 * random.random() + 0.7
            size_factor_v = 0.6 * random.random() + 0.7
            resize_matrix = torch.tensor([[size_factor_h, 0.],
                                          [0., size_factor_v]])
            transform_matrix = torch.mm(transform_matrix, resize_matrix)

        # First random translates ids, then apply matrix
        effective_sizes = torch.max(tiles_locations, dim=0)[0] - torch.min(tiles_locations, dim=0)[0]
        random_indexes = [random.randint(0, int(size)) for size in effective_sizes]
        translation_matrix = torch.tensor(random_indexes)
        tiles_locations -= translation_matrix.to(device)
        # Applies transformation
        tiles_locations = torch.mm(tiles_locations.float(), transform_matrix.to(device))

        # Offsets tiles to the leftmost and rightmost
        tiles_locations -= torch.min(tiles_locations, dim=0, keepdim=True)[0]
        return tiles_locations

    def forward(self, x, tiles_original_locations):

        x = torch.concat(x)
        x = x.cuda()
        
        x = self.adapt_layer(x)

        if self.perf_aug:
            tiles_locations = batched_coordinates(
                [self.data_augment_tiles_locations(tl / self.sparse_map_downsample)
                 for tl in tiles_original_locations])
        else:
            tiles_locations = batched_coordinates(
                [tl / self.sparse_map_downsample for tl in tiles_original_locations])

        tiles_locations = tiles_locations.to(x.device)

        sparse_map = SparseTensor(features=x,
                                  coordinates=tiles_locations,
                                  quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            dense_map = sparse_map.dense()[0]

        if dense_map.shape[-1] < self.min_size[1] or dense_map.shape[-2] < self.min_size[0]:
            pad = [max((self.min_size[1] - dense_map.shape[-1]) // 2, 0),
                   max((self.min_size[1] - dense_map.shape[-1]) // 2 + 1, 0),
                   max((self.min_size[0] - dense_map.shape[-2]) // 2, 0),
                   max((self.min_size[0] - dense_map.shape[-2]) // 2 + 1, 0)]

            dense_map = F.pad(dense_map, pad, mode='constant', value=0)
        #dense_map = self.adapt_layer(dense_map)
        dense_map = self.model(dense_map)
        result = self.pool(dense_map)
        result = result.view(result.size(0), result.size(1))
        result = self.classifier(result)

        if torch.isnan(result).any():
            to_save = {"x": x.cpu(), "locations": tiles_locations.cpu(),
                       "adapt_layer": self.adapt_layer.cpu(), "model": self.model.cpu()}
            torch.save(to_save, "./debug_nan.pt")
            print("NaN predictions")
            exit()

        return result, tiles_locations.cpu()
