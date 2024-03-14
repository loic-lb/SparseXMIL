import pandas as pd
import numpy as np

'''
initiate a pandas df describing a list of slides to process
args:
	slides (df or array-like): 
		array-like structure containing list of slide ids, if df, these ids assumed to be
		stored under the 'slide_id' column
	seg_params (dict): segmentation paramters 
	filter_params (dict): filter parameters
	vis_params (dict): visualization paramters
	patch_params (dict): patching paramters
	use_heatmap_args (bool): whether to include heatmap arguments such as ROI coordinates
'''


def initialize_df(slides, seg_params, filter_params, vis_params, patch_params,
                  use_heatmap_args=False, save_patches=False):
    total = len(slides)
    if isinstance(slides, pd.DataFrame):
        slide_ids = slides.slide_id.values
    else:
        slide_ids = slides
    default_df_dict = {'slide_id': slide_ids, 'process': np.full((total), 1, dtype=np.uint8)}

    # initiate empty labels in case not provided
    if use_heatmap_args:
        default_df_dict.update({'label': np.full((total), -1)})

    default_df_dict.update({
        'status': np.full((total), 'tbp'),
        # seg params
        'seg_level': np.full((total), int(seg_params['seg_level']), dtype=int),
        'window_avg': np.full((total), int(seg_params['window_avg']), dtype=int),
        'window_eng': np.full((total), int(seg_params['window_eng']), dtype=int),
        'thresh': np.full((total), int(seg_params['thresh']), dtype=int),

        # filter params
        'area_min': np.full((total), int(filter_params['area_min']), dtype=np.float32),

        # vis params
        'vis_level': np.full((total), int(vis_params['vis_level']), dtype=np.int8),
        'line_thickness': np.full((total), int(vis_params['line_thickness']), dtype=np.uint32),

        # patching params
        'use_padding': np.full((total), bool(patch_params['use_padding']), dtype=bool),
        'contour_fn': np.full((total), patch_params['contour_fn'])
    })

    if save_patches:
        default_df_dict.update({
            'white_thresh': np.full((total), int(patch_params['white_thresh']), dtype=np.uint8),
            'black_thresh': np.full((total), int(patch_params['black_thresh']), dtype=np.uint8)})

    if use_heatmap_args:
        # initiate empty x,y coordinates in case not provided
        default_df_dict.update({'x1': np.empty((total)).fill(np.NaN),
                                'x2': np.empty((total)).fill(np.NaN),
                                'y1': np.empty((total)).fill(np.NaN),
                                'y2': np.empty((total)).fill(np.NaN)})

    if isinstance(slides, pd.DataFrame):
        temp_copy = pd.DataFrame(default_df_dict)  # temporary dataframe w/ default params
        # find key in provided df
        # if exist, fill empty fields w/ default values, else, insert the default values as a new column
        for key in default_df_dict.keys():
            if key in slides.columns:
                mask = slides[key].isna()
                slides.loc[mask, key] = temp_copy.loc[mask, key]
            else:
                slides.insert(len(slides.columns), key, default_df_dict[key])
    else:
        slides = pd.DataFrame(default_df_dict)

    return slides
