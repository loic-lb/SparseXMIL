import os
import random
import h5py
import pandas as pd
import numpy as np
import multiprocessing

import torch.utils.data.dataset


class DatasetPretrained(torch.utils.data.dataset.Dataset):
    def __init__(self, patches_folder, feats_folder, slides_label_filepath, n_sampled_tiles_per_slide=0,
                 level=0, tile_size=(256, 256), n_workers=1, percentage=None, sort_tiles=False):

        self.percentage = percentage

        self.patches_folder = patches_folder
        assert os.path.exists(patches_folder)
        self.feats_folder = feats_folder
        assert os.path.exists(feats_folder)
        self.slides_label_filepath = slides_label_filepath
        assert os.path.exists(slides_label_filepath)

        self.n_workers = n_workers

        self.level = level
        self.tile_size = tile_size

        self.n_sampled_tiles_per_slide = n_sampled_tiles_per_slide

        self.sort_tiles = sort_tiles

        self.slides_feats, self.tiles_locations, self.slides_ids = self.load_tile_paths()
        self.slides_labels, self.correspondence_digit_label_name = self.load_labels()
        self.n_classes = len(self.correspondence_digit_label_name)

    @staticmethod
    def read_h5_file(hdf5_file_path):
        """
        Reads a h5 file and returns the list of corresponding coordinates
        @param hdf5_file_path: path to the h5 file
        @return: list of coordinates
        """
        file = h5py.File(hdf5_file_path, 'r')
        coords = file["coords"]
        return list(coords)

    def load_tile_paths(self):
        """
        Retrieves the paths to the features files, the slide ids, and loads the corresponding tiles coordinates
        @return: paths to the features files, tiles coordinates, slide ids
        """
        slides_feats = np.array([os.path.join(self.feats_folder, file) for file in os.listdir(self.feats_folder)
                                 if file.endswith(".pt")])
        slides_feats.sort()

        # Get absolute patch files
        tiles_paths = np.array([os.path.join(self.patches_folder, os.path.basename(file).replace(".pt", ".h5")) for file
                                in slides_feats])

        slides_ids = np.array([os.path.basename(file).replace(".pt", "") for file in slides_feats])

        # Recovers tiles coordinates
        if self.n_workers > 1:
            pool = multiprocessing.Pool(self.n_workers)
            tiles_locations = pool.map(self.read_h5_file, tiles_paths)
            pool.close()
            pool.join()
        else:
            tiles_locations = []
            for hdf5_file_path in tiles_paths:
                file = h5py.File(hdf5_file_path, 'r')
                coords = file["coords"]
                tiles_locations.append(list(coords))

        tiles_locations = np.array(tiles_locations, dtype="object")

        return slides_feats, tiles_locations, slides_ids

    def load_labels(self):
        """
        Loads the labels of the slides
        @return: labels of the slides as a digit, correspondence between digits and labels
        """
        csv_filepath = self.slides_label_filepath
        df = pd.read_csv(csv_filepath)
        slides_ids = df.slide_id.values
        slides_labels = df.label.values
        sorted_index_tiles = np.argsort(slides_ids)
        slides_ids = slides_ids[sorted_index_tiles]
        slides_labels = slides_labels[sorted_index_tiles]

        # Converts labels to digits
        unique_labels = list(set(slides_labels))
        unique_labels.sort()
        assert len(unique_labels) > 1, f'Expected at least two labels, found {len(unique_labels)}'
        label_to_digit = {label: digit for digit, label in enumerate(unique_labels)}
        digit_to_label = {digit: label for digit, label in enumerate(unique_labels)}
        slides_labels = {slide_id: label_to_digit[label] for slide_id, label in zip(slides_ids, slides_labels)}

        return slides_labels, digit_to_label

    def __getitem__(self, slide_index):
        """
        Loads the features of a fixed number of tiles of a slide (if n_sampled_tiles_per_slide is set) or a percentage
        of the tiles of a slide (if percentage is set) or all tiles of a slide (if n_sampled_tiles_per_slide and
        percentage are not set) and their locations, the label of the slide and its id
        @param slide_index: index of the slide to retrieve
        @return: sampled tiles features, sampled tiles locations, slide label, slide id
        """
        slide_feats = torch.load(self.slides_feats[slide_index])
        slide_tiles_locations = np.array(self.tiles_locations[slide_index])

        n_tiles = len(slide_tiles_locations)
        if self.percentage:
            if self.sort_tiles:
                sampled_tiles_indexes = sorted(random.sample(range(n_tiles), k=int(self.percentage * n_tiles)))
            else:
                sampled_tiles_indexes = random.sample(range(n_tiles), k=int(self.percentage * n_tiles))
        elif self.n_sampled_tiles_per_slide < 1:
            sampled_tiles_indexes = np.arange(n_tiles)
        else:
            if self.sort_tiles:
                sampled_tiles_indexes = sorted(random.sample(range(n_tiles), k=self.n_sampled_tiles_per_slide))
            else:
                sampled_tiles_indexes = random.sample(range(n_tiles), k=self.n_sampled_tiles_per_slide)
        # Loads all tiles and their locations
        sampled_locations = slide_tiles_locations[sampled_tiles_indexes]
        sampled_tiles_feats = slide_feats[sampled_tiles_indexes]

        tiles_locations = torch.tensor(sampled_locations)

        # Load associated slide label
        slide_id = self.slides_ids[slide_index]
        slide_label = self.slides_labels[slide_id]

        return sampled_tiles_feats, tiles_locations, slide_label, slide_id

    def __len__(self):
        # Length is set to the number of slides
        return len(self.slides_ids)
