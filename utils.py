import cv2
import random
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import default_collate
from torchvision.utils import make_grid
from skimage.color import rgb2gray
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, f1_score
from sklearn.metrics import auc as calc_auc
from scipy.stats import rankdata
from matplotlib.colors import LinearSegmentedColormap
from MinkowskiEngine.utils import batched_coordinates
from MinkowskiEngine import (SparseTensor,
                             to_sparse,
                             SparseTensorQuantizationMode)

cm = LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:000000-10:69ECEE-20:F0FF00-30:F6802E-40:FF000C
    (0.000, (0.000, 0.000, 0.000)),
    (0.100, (0.412, 0.925, 0.933)),
    (0.200, (0.941, 1.000, 0.000)),
    (0.300, (0.965, 0.502, 0.180)),
    (0.400, (1.000, 0.000, 0.047)),
    (1.000, (1.000, 0.000, 0.047))))


def custom_collate(batch):
    """
    Custom collate function for the dataloader
    @param batch: a batch from dataloader
    @return: list of tiles embeddings, list of tiles locations, a batch tensor labels, and a batch tensor slide ids
    """
    it = iter(batch)
    elem_size = len(next(it))
    if not all(len(elem) == elem_size for elem in it):
        raise RuntimeError('each element in list of batch should be of equal size')
    transposed = list(zip(*batch))
    return [list(transposed[0]),  # tiles
            list(transposed[1]),  # tiles locations
            default_collate(transposed[2]),  # labels
            default_collate(transposed[3])]  # slide ids


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(dataset, batch_size, shuffle, num_workers, custom_collate_bool=True, seed=0):
    """
    Creates a dataloader with the specified parameters
    @param dataset: dataset to use for the dataloader
    @param batch_size: batch size
    @param shuffle: set to True to shuffle the dataset
    @param num_workers: number of workers
    @param custom_collate_bool: set to True to use the custom collate function
    @param seed: seed for the workers
    @return: a dataloader with the specified parameters
    """
    custom_collate_fn = custom_collate if custom_collate_bool else default_collate
    if num_workers > 0:
        g = torch.Generator()
        g.manual_seed(seed)
        return torch.utils.data.DataLoader(dataset, batch_size, shuffle, num_workers=num_workers,
                                           collate_fn=custom_collate_fn,
                                           worker_init_fn=seed_worker,
                                           generator=g)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size, shuffle, num_workers=num_workers,
                                           collate_fn=custom_collate_fn)


def split_dataset(dataset, test_size, split=None):
    """
    Splits the dataset into a train and a test set
    @param dataset: dataset to split
    @param test_size: size of the test set (when split is not specified)
    @param split: path to the csv file containing the split (train, val) of the dataset (with slide ids)
    @return: two subsets of the dataset, one for training and one for validation
    """
    if split:
        split_csv = pd.read_csv(split)
        train_indices = np.nonzero(np.in1d(dataset.slides_ids, split_csv.train.values))[0]
        if "val" not in split_csv.columns:
            train_indices, test_indices = train_test_split(train_indices, test_size=test_size,
                                                           stratify=[dataset.slides_labels[dataset.slides_ids[idx]] for
                                                                     idx in train_indices],
                                                           random_state=0)
        else:
            test_indices = np.nonzero(np.in1d(dataset.slides_ids, split_csv.val.values))[0]
    else:
        dataset_size = len(dataset)
        indices = np.arange(dataset_size)
        train_indices, test_indices = train_test_split(indices, test_size=test_size,
                                                       stratify=list(dataset.slides_labels.values()), random_state=0)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    print(f'Train size: {len(train_dataset)}')
    print(f'Test size: {len(test_dataset)}')

    return train_dataset, test_dataset


def create_illustrations(tiles_locations, as_list=False):
    """
    Creates an illustration of the tiles locations for each batch
    @param tiles_locations: tiles locations in Minkowski format (batch_idx, x, y)
    @param as_list: return a list of illustrations if True, otherwise a tensor grid
    @return: a tensor grid or a list of tiles locations projected on a sparse map
    """
    illustrations = []
    tiles_locations = tiles_locations
    for batch_idx in torch.unique(tiles_locations[:, 0]):
        tiles_locations_map = tiles_locations[tiles_locations[:, 0] == batch_idx][:, 1:]
        sparse_map = torch.zeros((int(tiles_locations[:, 1].max()) + 1, int(tiles_locations[:, 2].max()) + 1),
                                 dtype=int)
        sparse_map[tiles_locations_map[:, 0].long(), tiles_locations_map[:, 1].long()] = torch.ones(
            tiles_locations_map.shape[0],
            dtype=int)
        illustrations.append(sparse_map.unsqueeze(0))
    if as_list:
        return illustrations
    else:
        illustrations = torch.stack(illustrations)
        grid = np.transpose(make_grid(illustrations).numpy(), (1, 2, 0))
        return (rgb2gray(grid) > 0) * 255


def apply_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def measure_perf(losses, ground_truths, predicted_classes, probas, n_classes=2):
    """
    Computes the loss, balanced accuracy, f1 score and auc
    @param losses: list of loss values
    @param ground_truths: list of ground truth labels
    @param predicted_classes: list of predicted classes
    @param probas: list of predicted probabilities
    @param n_classes: number of classes
    @return: average values of loss, balanced accuracy, f1 score and auc
    """
    loss = np.mean(losses)
    bac = balanced_accuracy_score(ground_truths, predicted_classes)
    f1 = f1_score(ground_truths, predicted_classes, average="weighted")
    if n_classes == 2:
        auc = roc_auc_score(ground_truths, probas[:, 1])
    else:
        aucs = []
        binary_labels = label_binarize(ground_truths, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in ground_truths:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], probas[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
        auc = np.nanmean(np.array(aucs))
    return loss, bac, f1, auc


class FeatureExtractor:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                if isinstance(x, SparseTensor):
                    x = x.dense()[0]
                    x.register_hook(self.save_gradient)
                    outputs += [x]
                    x = to_sparse(x)
        return outputs, x


class ModelOutputs:
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermediate targeted layers.
    3. Gradients from intermediate targeted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x, tiles_ids=None):
        target_activations = []
        for name_module, module in self.model._modules.items():
            if "sparse_model" in name_module:
                for name_submodule, submodule in module._modules.items():
                    if submodule == self.feature_module:
                        target_activations, x = self.feature_extractor(x)
                    else:
                        x = submodule(x)
            # target_activations, x = self.feature_extractor(x.squeeze(0))
            else:
                x = module(x)
        return target_activations, x.F


class GradCAM:
    """
    Produces class activation map using the gradient information flowing into the target layer for the predicted class,
    and weights the activation maps by the average gradient values of each feature map.
    """

    def __init__(self, model, feature_module, target_layer_names):

        self.model = model
        self.feature_module = feature_module

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

        self.target_size = None

    def set_target_size(self, target_size):
        self.target_size = target_size

    def __call__(self, input_img, target_category=None, extractor=None):

        features, output = self.extractor(input_img)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-12)
        return cam


def create_heatmap_from_cam(gradcam, x, locations, sparse_map_downsample):
    """
    Creates a heatmap from sparse model using GradCAM
    @param gradcam: GradCAM object
    @param x: tiles embeddings
    @param locations: tiles locations
    @param sparse_map_downsample: downsampling factor of the sparse map
    @return: heatmap for output class using GradCAM algorithm
    """
    x = x.cuda()

    tiles_locations = batched_coordinates([tl / sparse_map_downsample for tl in locations])
    tiles_locations = tiles_locations.to(x.device)
    sparse_map = SparseTensor(features=x,
                              coordinates=tiles_locations,
                              quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
    return gradcam(sparse_map)


def create_heatmap_from_attention(attention_weights, locations, sparse_map_downsample,
                                  patch_size, patch_scale_factor=1, transmil=False):
    """
    Creates a heatmap from attention-based model using attention weights
    @param attention_weights: attention weights from attention-based model
    @param locations: tiles locations
    @param sparse_map_downsample: sparse map downsample
    @param patch_size: patch size used to produce the embeddings
    @param patch_scale_factor: a scale factor to increase the size of the patches (create overlap between patches)
    @param transmil: set to True if the model is TransMIL
    @return: heatmap for output class using attention weights
    """
    if transmil:
        H = len(locations)
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        N = _H * _W
        attention_weights = attention_weights[0][..., -(N + 1):, -(N + 1):]
        scores = attention_weights.mean(axis=1)[:, 0, 1: H + 1].squeeze().cpu().detach().numpy()
    else:
        scores = attention_weights[0].squeeze().cpu().detach().numpy()
    scores = rankdata(scores, "average") / len(scores)
    locations_downsample = (locations / sparse_map_downsample).int()
    grayscale_attention_map = torch.zeros((np.ceil(locations_downsample.max(axis=0)[0])).int().tolist())
    grid_size = int(np.ceil(patch_size / sparse_map_downsample))
    patch_size = int(np.ceil(patch_size / sparse_map_downsample * patch_scale_factor))
    for k, loc in enumerate(locations_downsample):
        if loc[0] * grid_size + patch_size < grayscale_attention_map.shape[0] and loc[1] * grid_size + patch_size \
                < grayscale_attention_map.shape[1]:
            grayscale_attention_map[loc[0] * grid_size: loc[0] * grid_size + patch_size,
            loc[1] * grid_size:loc[1] * grid_size + patch_size] = torch.ones((patch_size, patch_size)) * scores[k]
    grayscale_attention_map = grayscale_attention_map.numpy()
    return grayscale_attention_map


def create_rgb_heatmap(grayscale_heatmap):
    """
    Creates a RGB heatmap from a grayscale heatmap
    @param grayscale_heatmap: input grayscale heatmap
    @return: a RGB heatmap with cm custom colormap applied
    """
    color_range = (cm(range(256)) * 255).astype("uint8")[:, :3]
    color_range = np.squeeze(np.dstack([color_range[:, 2], color_range[:, 1], color_range[:, 0]]), 0)
    channels = [cv2.LUT(np.uint8(255 * grayscale_heatmap), color_range[:, i]) for i in range(3)]
    channels = np.dstack(channels)
    return channels


def create_mask_from_contours(WSI_object, heatmap_shape, downsample):
    """
    Creates a mask from the contours of the tissue with the same shape as the heatmap (for interpretation code)
    @param WSI_object: WholeSlideImage object
    @param heatmap_shape: Shape of the heatmap
    @param downsample: downsample factor corresponding to the visualization level of the heatmap
    @return: a binary mask of the tissue
    """
    seg_params = {"seg_level": 3, "window_avg": 30, "window_eng": 3, "thresh": 90}
    filter_params = {'area_min': 3e3}
    scale = [1 / downsample, 1 / downsample]
    if len(WSI_object.level_dim) <= seg_params["seg_level"]:
        seg_params["seg_level"] = len(WSI_object.level_dim) - 1
    WSI_object.segmentTissue(**seg_params, area_min=filter_params["area_min"])
    mask = np.zeros(heatmap_shape)
    cv2.drawContours(mask, WSI_object.scaleContourDim(WSI_object.contours_tissue, scale), -1, (1), -1)
    mask = np.expand_dims(mask, -1).repeat(3, -1).astype("uint8")
    return mask
