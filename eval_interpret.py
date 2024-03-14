import os
import argparse
import cv2
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve
from sklearn.model_selection import train_test_split
from utils import apply_random_seed
from tqdm import tqdm


def remove_blue_annotations(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 200, 200])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    img[mask > 0] = [0, 0, 0]
    return img


def main():
    parser = argparse.ArgumentParser(description="Compute performance of attended areas compared to annotations")
    parser.add_argument("--heatmaps_path", type=str, help="path to heatmap produced by interpret.py")
    parser.add_argument("--annotations_path", type=str, help="path to annotation folder")
    parser.add_argument("--masks_path", type=str, help="path to tissue's masks folder")
    parser.add_argument("--validation_split", type=float, default=0.2, help="validation split")
    parser.add_argument("--seed", type=int, default=512, help="random seed")

    args = parser.parse_args()
    apply_random_seed(args.seed)

    slide2annotation_file_path = os.path.join(args.annotations_path, "slide_list.txt")
    df_slide2annotation = pd.read_csv(slide2annotation_file_path, sep=" ", header=None)
    slide2annotation = dict(zip(df_slide2annotation[2].str.replace(".svs", ""), df_slide2annotation[1]))
    annotated_slides_file_path = os.path.join(args.annotations_path, "slide_list.csv")
    labeled_slides = pd.read_csv(annotated_slides_file_path).slide_id.values

    validation_slides_id, testing_slides_id = train_test_split(labeled_slides, train_size=0.1, random_state=args.seed)

    validation_annot_imgs = []
    validation_heatmap_imgs = []
    for slide_id in tqdm(validation_slides_id):
        heatmap_file_path = os.path.join(args.heatmaps_path, f"{slide_id}_heatmap.npy")
        heatmap_img = np.load(heatmap_file_path)
        annotation_path = os.path.join(args.annotations_path, "complete_region_annotation",
                                       f"{slide2annotation[slide_id]}.png")
        annot_img = cv2.imread(annotation_path)
        if annot_img.shape[0] > 2 * heatmap_img.shape[0] and annot_img.shape[1] > 2 * heatmap_img.shape[1]:
            print("problem")
            break
        elif annot_img.shape[0] < heatmap_img.shape[0] or annot_img.shape[1] < heatmap_img.shape[1]:
            print("problem")
            break
        annot_img = remove_blue_annotations(annot_img)[: heatmap_img.shape[0], : heatmap_img.shape[1], :]
        mask_path = os.path.join(args.masks_path, f"{slide_id}_mask.npy")
        mask_contours = np.load(mask_path)[..., 0]
        # annot_img, heatmap_img = process_imgs(annot_img, heatmap_img)
        validation_annot_imgs.append((cv2.cvtColor(annot_img, cv2.COLOR_BGR2GRAY) >
                                      0).astype(np.uint8).ravel()[mask_contours.ravel() > 0])
        validation_heatmap_imgs.append(heatmap_img.ravel()[mask_contours.ravel() > 0])

    precision, recall, thresholds = precision_recall_curve(np.concatenate(validation_annot_imgs).ravel(),
                                                           np.concatenate(validation_heatmap_imgs).ravel())
    eps = 1e-12
    fscore = (2 * precision * recall) / (precision + recall + eps)
    ix = np.argmax(fscore)
    print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))

    aucs = []
    bacs = []
    precisions = []
    recalls = []
    f1s = []
    for slide_id in tqdm(testing_slides_id):
        heatmap_file_path = os.path.join(args.heatmaps_path, f"{slide_id}_heatmap.npy")
        heatmap_img = np.load(heatmap_file_path)
        annotation_path = os.path.join(args.annotations_path, "complete_region_annotation",
                                       f"{slide2annotation[slide_id]}.png")
        annot_img = cv2.imread(annotation_path)
        annot_img = remove_blue_annotations(annot_img)[: heatmap_img.shape[0], : heatmap_img.shape[1], :]
        annot_img = (cv2.cvtColor(annot_img, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
        mask_path = os.path.join(args.masks_path, f"{slide_id}_mask.npy")
        mask_contours = np.load(mask_path)[..., 0]

        heatmap_img = heatmap_img.ravel()[mask_contours.ravel() > 0]
        annot_img = annot_img.ravel()[mask_contours.ravel() > 0]

        aucs.append(roc_auc_score(annot_img, heatmap_img))
        heatmap_img = (heatmap_img > thresholds[ix]).astype(np.uint8)
        bacs.append(balanced_accuracy_score(annot_img, heatmap_img))
        precisions.append(precision_score(annot_img, heatmap_img))
        recalls.append(recall_score(annot_img, heatmap_img))
        f1s.append(f1_score(annot_img, heatmap_img))

    print(f"AUC: {np.mean(aucs)}")
    print(f"Balanced accuracy: {np.mean(bacs)}")
    print(f"Precision: {np.mean(precisions)}")
    print(f"Recall: {np.mean(recalls)}")
    print(f"F1: {np.mean(f1s)}")


if __name__ == "__main__":
    main()
