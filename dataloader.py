from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
from config import *
import tifffile


class CustomImageDataset(Dataset):
    def __init__(self, dataframe, augmenter_bool, path_contour_gt, path_gt, path_image):
        self.path_contour_gt = path_contour_gt
        self.path_gt = path_gt
        self.path_image = path_image
        self.dataframe = dataframe
        self.indices = self.dataframe.index.tolist()
        self.augmenter_bool = augmenter_bool

    def __len__(self):
        return len(self.dataframe)

    def augmenter(self, image, output):
        k = np.random.choice([1, 2, 3])

        image = np.rot90(image, k=k, axes=(0, 1))
        output = np.rot90(output, k=k, axes=(0, 1))
        alea_shift1 = np.random.random()
        alea_shift2 = np.random.random()

        if alea_shift1 > 0.5:
            image = np.flipud(image)
            output = np.flipud(output)

        if alea_shift2 > 0.5:
            image = np.fliplr(image)
            output = np.fliplr(output)

        return image, output

    def __getitem__(self, idx):
        filename = self.dataframe.iloc[idx]["filename"]
        image = tifffile.imread(os.path.join(self.path_image, filename)) / 255
        """ Ground truth """

        contour_gt = tifffile.imread(
            os.path.join(self.path_contour_gt, filename)
        ).astype(float)
        patch_gt = (tifffile.imread(os.path.join(self.path_gt, filename)) > 0).astype(
            float
        )

        patch_gt = ((patch_gt - contour_gt) > 0).astype(float)
        output = np.expand_dims(
            np.zeros((parameters["dim"], parameters["dim"]))
            + contour_gt
            + 2 * patch_gt,
            -1,
        )

        """ Augmenter """

        if self.augmenter_bool:
            image, output = self.augmenter(image, output)

        image = np.array(np.transpose(image, (2, 0, 1)), dtype=np.float32)
        output = np.array(np.transpose(output, (2, 0, 1)), dtype=np.float32)

        return (image, output)


dataset_train = CustomImageDataset(
    path_image=path_images,
    path_contour_gt=path_contour_gt,
    path_gt=path_gt,
    dataframe=df_train,
    augmenter_bool=True,
)


dataset_test = CustomImageDataset(
    path_image=path_images,
    path_contour_gt=path_contour_gt,
    path_gt=path_gt,
    dataframe=df_test,
    augmenter_bool=False,
)


dataset_val = CustomImageDataset(
    path_image=path_images,
    path_contour_gt=path_contour_gt,
    path_gt=path_gt,
    dataframe=df_val,
    augmenter_bool=False,
)

loader_train = DataLoader(
    batch_size=parameters["batch_size"],
    dataset=dataset_train,
    num_workers=16,
    shuffle=True,
)

loader_val = DataLoader(
    batch_size=parameters["batch_size"],
    dataset=dataset_val,
    num_workers=16,
    shuffle=False,
)

loader_test = DataLoader(
    batch_size=parameters["batch_size"],
    dataset=dataset_test,
    num_workers=16,
    shuffle=False,
)

dataloaders = {"train": loader_train, "test": loader_test, "val": loader_val}
