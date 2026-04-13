# viewer.py
# Run this file to explore any MedMNIST dataset

from utils import (
    load_dataset,
    dataset_to_arrays,
    get_dataset_info,
    show_label_distribution,
    show_samples,
    interactive_viewer
)


# change data_flag to explore different datasets
data_flag = "bloodmnist"
# "adrenalmnist3d"
# "bloodmnist"
# "breastmnist"
# "chestmnist"
# "dermamnist"
# "fracturemnist3d"
# "nodulmnist3d"
# "octmnist"
# "organamnist"
# "organcmnist"
# "organmnist3d"
# "organsmnist"
# "pathmnist"
# "pneumoniamnist"
# "retinamnist"
# "synapsemnist3d"
# "tissuemnist"
# "vesselmnist3d"


def main():
    print(f"\nLoading dataset: {data_flag}")

    # Load dataset
    train_ds, val_ds, test_ds = load_dataset(data_flag)

    # Convert to numpy
    X_train, y_train = dataset_to_arrays(train_ds, "train", data_flag)

    # Get dataset info
    info = get_dataset_info(data_flag)

    # Show label distribution
    show_label_distribution(y_train, info)

    # Show sample grid
    show_samples(X_train, y_train, info, num_samples=10)

    # Interactive viewer
    interactive_viewer(X_train, y_train, info)


if __name__ == "__main__":
    main()