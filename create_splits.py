import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


def save_split(dataset, train_ids, val_ids, test_ids, filename):
    print("Train split composition:")
    value_counts = dataset.loc[train_ids, 'label'].value_counts()
    for category, count in value_counts.items():
        print(f"{category}: {count}")
    train_split = dataset.loc[train_ids, 'slide_id']
    val_split = dataset.loc[val_ids, 'slide_id']
    print("Val split composition:")
    value_counts = dataset.loc[val_ids, 'label'].value_counts()
    for category, count in value_counts.items():
        print(f"{category}: {count}")
    test_split = dataset.loc[test_ids, 'slide_id']
    print("Test split composition:")
    value_counts = dataset.loc[test_ids, 'label'].value_counts()
    for category, count in value_counts.items():
        print(f"{category}: {count}")
    df_tr = pd.DataFrame({'train': train_split}).reset_index(drop=True)
    df_v = pd.DataFrame({'val': val_split}).reset_index(drop=True)
    df_t = pd.DataFrame({'test': test_split}).reset_index(drop=True)
    df = pd.concat([df_tr, df_v, df_t], axis=1)
    df.to_csv(filename, index=False)


def main():
    parser = argparse.ArgumentParser(description="Create splits for given dataset")
    parser.add_argument('--dataset', type=str, default='./dataset.csv', help='dataset to use')
    parser.add_argument('--output', type=str, default='./splits', help='output directory')
    parser.add_argument('--n_splits', type=int, default=10, help='number of splits')
    parser.add_argument('--val_frac', type=float, default=0.1, help='fraction of validation data')
    parser.add_argument('--test_frac', type=float, default=0.1, help='fraction of test data')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(args.dataset)
    num_classes = len(dataset.label.unique())
    print(f'Number of classes: {num_classes}')

    patients = np.unique(np.array(dataset['case_id']))  # get unique patients
    patient_labels = []
    for p in patients:
        locations = dataset[dataset['case_id'] == p].index.tolist()
        assert len(locations) > 0
        label = dataset['label'][locations].values
        label = stats.mode(label, keepdims=True)[0]
        patient_labels.append(label)
    patient_data = {'case_id': patients, 'label': np.array(patient_labels)}

    cls_ids = [[] for i in range(num_classes)]
    for i in range(num_classes):
        cls_ids[i] = np.where(patient_data['label'] == i)[0]

    rng = np.random.default_rng(args.seed)
    num_slides_cls = np.array([len(cls_id) for cls_id in cls_ids])
    val_num = np.round(num_slides_cls * args.val_frac).astype(int)
    test_num = np.round(num_slides_cls * args.test_frac).astype(int)
    samples = len(patients)
    indices = np.arange(samples).astype(int)

    def create_splits():
        for i in range(args.n_splits):
            all_val_ids = []
            all_test_ids = []
            sampled_train_ids = []
            for c in range(len(val_num)):
                possible_indices = np.intersect1d(cls_ids[c], indices)  # all indices of this class
                val_ids = rng.choice(list(possible_indices), val_num[c], replace=False)  # validation ids

                remaining_ids = np.setdiff1d(possible_indices, val_ids)  # indices of this class left after validation
                all_val_ids.extend(val_ids)

                test_ids = rng.choice(remaining_ids, test_num[c], replace=False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

                sampled_train_ids.extend(remaining_ids)
            yield sampled_train_ids, all_val_ids, all_test_ids

    generated_splits = create_splits()
    for k in range(args.n_splits):
        print(f"Split {k}:")
        ids = next(generated_splits)
        slide_ids = [[] for i in range(len(ids))]

        for split in range(len(ids)):
            for idx in ids[split]:
                case_id = patient_data['case_id'][idx]
                slide_indices = dataset[dataset['case_id'] == case_id].index.tolist()
                slide_ids[split].extend(slide_indices)

        train_ids, val_ids, test_ids = slide_ids[0], slide_ids[1], slide_ids[2]
        save_split(dataset, train_ids, val_ids, test_ids, f'{args.output}/split_{k}.csv')


if __name__ == '__main__':
    main()
