import h5py
import numpy as np
from scipy.signal import oaconvolve
from PIL import Image
import math
import cv2


def isWhitePatch(patch, satThresh=5):
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    return True if np.mean(patch_hsv[:, :, 1]) < satThresh else False


def isBlackPatch(patch, rgbThresh=40):
    return True if np.all(np.mean(patch, axis=(0, 1)) < rgbThresh) else False


def isBlackPatch_S(patch, rgbThresh=20, percentage=0.05):
    num_pixels = patch.size[0] * patch.size[1]
    return True if np.all(np.array(patch) < rgbThresh, axis=(2)).sum() > num_pixels * percentage else False


def isWhitePatch_S(patch, rgbThresh=220, percentage=0.2):
    num_pixels = patch.size[0] * patch.size[1]
    return True if np.all(np.array(patch) > rgbThresh, axis=(2)).sum() > num_pixels * percentage else False


def save_hdf5(output_path, asset_dict, attr_dict=None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path


def DrawGrid(img, coord, shape, thickness=2, color=(0, 0, 0, 255)):
    cv2.rectangle(img, tuple(np.maximum([0, 0], coord - thickness // 2)),
                  tuple(coord - thickness // 2 + np.array(shape)), (0, 0, 0, 255), thickness=thickness)
    return img


def DrawMapFromCoords(canvas, wsi_object, coords, patch_size, vis_level, indices=None, verbose=1, draw_grid=True):
    downsamples = wsi_object.wsi.level_downsamples[vis_level]
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)

    patch_size = tuple(np.ceil((np.array(patch_size) / np.array(downsamples))).astype(np.int32))
    print('downscaled patch size: {}x{}'.format(patch_size[0], patch_size[1]))

    for idx in range(total):
        if verbose > 0:
            if idx % ten_percent_chunk == 0:
                print('progress: {}/{} stitched'.format(idx, total))

        patch_id = indices[idx]
        coord = coords[patch_id]
        patch = np.array(wsi_object.wsi.read_region(tuple(coord), vis_level, patch_size).convert("RGB"))
        patch
        coord = np.ceil(coord / downsamples).astype(np.int32)
        canvas_crop_shape = canvas[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0], :3].shape[:2]
        canvas[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0], :3] = patch[:canvas_crop_shape[0],
                                                                                           :canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, patch_size)

    return Image.fromarray(canvas)


def StitchCoords(hdf5_file_path, wsi_object, downscale=16, draw_grid=False, bg_color=(0, 0, 0), alpha=-1):
    wsi = wsi_object.wsi
    vis_level = wsi.get_best_level_for_downsample(downscale)
    file = h5py.File(hdf5_file_path, 'r')
    dset = file['coords']
    coords = dset[:]
    w, h = wsi.level_dimensions[0]

    print('start stitching {}'.format(dset.attrs['name']))
    print('original size: {} x {}'.format(w, h))

    w, h = wsi.level_dimensions[vis_level]

    print('downscaled size for stiching: {} x {}'.format(w, h))
    print('number of patches: {}'.format(len(coords)))

    patch_size = dset.attrs['patch_size']
    patch_level = dset.attrs['patch_level']
    print('patch size: {}x{} patch level: {}'.format(patch_size, patch_size, patch_level))
    patch_size = tuple((np.array((patch_size, patch_size)) * wsi.level_downsamples[patch_level]).astype(np.int32))
    print('ref patch size: {}x{}'.format(patch_size, patch_size))

    if w * h > Image.MAX_IMAGE_PIXELS:
        raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)

    if alpha < 0 or alpha == -1:
        heatmap = Image.new(size=(w, h), mode="RGB", color=bg_color)
    else:
        heatmap = Image.new(size=(w, h), mode="RGBA", color=bg_color + (int(255 * alpha),))

    heatmap = np.array(heatmap)
    heatmap = DrawMapFromCoords(heatmap, wsi_object, coords, patch_size, vis_level, indices=None, draw_grid=draw_grid)

    file.close()
    return heatmap


def remove_black_areas(slide):
    slidearr = np.asarray(slide).copy()
    indices = np.where(np.all(slidearr == (0, 0, 0), axis=-1))
    slidearr[indices] = [255, 255, 255]
    return Image.fromarray(slidearr)


def local_average(img, window_size, keep_grayscale):
    window = np.ones((window_size, window_size)) / (window_size ** 2)
    img_grayscaled = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #img_grayscaled = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)[..., 1]
    if keep_grayscale:
        return img_grayscaled
    else:
        return img_grayscaled - oaconvolve(img_grayscaled, window, mode='same')


def compute_law_feats(img, window_size):
    L5 = np.array([1, 4, 6, 4, 1])
    E5 = np.array([-1, -2, 0, 2, 1])
    S5 = np.array([-1, 0, 2, 0, -1])
    R5 = np.array([1, -4, 6, -4, 1])

    vectors = [L5, E5, S5, R5]
    filt = np.expand_dims(vectors[2], -1).dot(np.expand_dims(vectors[2], -1).T)

    img_filtered = oaconvolve(img, filt, mode="same")

    window = np.ones((window_size, window_size))
    img_energy = oaconvolve(np.abs(img_filtered), window, mode="same")

    return img_energy


def filter_ROI(roi):
    img = np.asarray(roi)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_filtered = cv2.bilateralFilter(img, 3, 3 * 2, 3 / 2)
    return img_filtered


def thresh_ROI(roi, thresh, inv):
    if inv:
        _, img_thresh = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY_INV)
    else:
        _, img_thresh = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)
    return img_thresh


def floodfill_ROI(roi, start):
    im_floodfill = roi.copy()
    h, w = roi.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, start, 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = roi | im_floodfill_inv
    return im_out[1:-1, 1:-1]


def contour_ROI(roi):
    contours = cv2.findContours(roi, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours
