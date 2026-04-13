# 2Dfullviewer.py
# Run this file to explore any MedMNIST dataset

from utils import (
    load_dataset,
    dataset_to_arrays,
    get_dataset_info,
    show_all_images_for_label
)


# change data_flag to explore different datasets
data_flag = "bloodmnist"
# "bloodmnist"
# "breastmnist"
# "chestmnist"
# "dermamnist"
# "octmnist"
# "organamnist"
# "organcmnist"
# "organsmnist"
# "pathmnist"
# "pneumoniamnist"
# "retinamnist"
# "tissuemnist"


def main():
    print(f"\nLoading dataset: {data_flag}")
    train_ds, val_ds, test_ds = load_dataset(data_flag)

    X_train, y_train = dataset_to_arrays(train_ds, "train", data_flag)

    info = get_dataset_info(data_flag)

    print("\nAvailable labels:")
    for k, v in info["label"].items():
        print(k, ":", v)

    label_id = int(input("\nEnter label ID to view: "))

    show_all_images_for_label(X_train, y_train, info, label_id)



if __name__ == "__main__":
    main()