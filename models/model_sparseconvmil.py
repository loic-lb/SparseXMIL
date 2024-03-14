import torch
import random
import math
import torch.nn as nn
from MinkowskiEngine import (MinkowskiGlobalAvgPooling,
                             MinkowskiConvolution,
                             MinkowskiLinear,
                             MinkowskiReLU,
                             SparseTensor,
                             MinkowskiBatchNorm,
                             MinkowskiMaxPooling)
from MinkowskiEngine import SparseTensorQuantizationMode
from MinkowskiEngine.utils import batched_coordinates


class SparseConvMIL(nn.Module):
    def __init__(self, nb_layers_in, sparse_map_downsample, perf_aug, num_classes, D=2,
                 tile_coordinates_rotation_augmentation=True, tile_coordinates_flips_augmentation=True,
                 tile_coordinates_resize_augmentation=True):
        super(SparseConvMIL, self).__init__()

        self.name = "SparseConvMIL"
        self.nb_layers_in = nb_layers_in
        self.sparse_map_downsample = sparse_map_downsample
        self.perf_aug = perf_aug
        self.num_classes = num_classes
        self.D = D
        self.tile_coordinates_rotation_augmentation = tile_coordinates_rotation_augmentation
        self.tile_coordinates_flips_augmentation = tile_coordinates_flips_augmentation
        self.tile_coordinates_resize_augmentation = tile_coordinates_resize_augmentation

        self.sparse_model = self.get_sparse_model(self.nb_layers_in)

        self.pool_minkow = MinkowskiGlobalAvgPooling()

        self.classifier = nn.Sequential(MinkowskiLinear(128, self.num_classes))

    def get_sparse_model(self, in_planes):
        return nn.Sequential(MinkowskiConvolution(in_planes, 128, kernel_size=3, stride=1, dimension=self.D),
                             MinkowskiReLU(),
                             MinkowskiConvolution(128, 128, kernel_size=3, stride=1, dimension=self.D),
                             MinkowskiReLU(),)

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

        if self.perf_aug:
            tiles_locations = batched_coordinates([self.data_augment_tiles_locations(tl / self.sparse_map_downsample)
                                                   for tl in tiles_original_locations])
        else:
            tiles_locations = batched_coordinates([tl / self.sparse_map_downsample for tl in tiles_original_locations])

        tiles_locations = tiles_locations.to(x.device)

        sparse_map = SparseTensor(features=x,
                                  coordinates=tiles_locations,
                                  quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

        sparse_map = self.sparse_model(sparse_map)

        result = self.pool_minkow(sparse_map)

        result = self.classifier(result).F

        return result, tiles_locations
