import os
import pandas as pd


path_weights = 'weights'
path_click_project = '/home/ahabis/1-Click_project'
path_pannuke = '/home/ahabis/sshfs/pannuke'
path_split = '/home/ahabis/1-Click_project/split_train_val_test'

path_gt = os.path.join(path_pannuke, 'gt')
path_contour_gt = os.path.join(path_pannuke, 'contour_gt')
path_binary_gt = os.path.join(path_pannuke, 'binary_gt')

path_fold1 = os.path.join(path_pannuke, 'fold_1')
path_fold2 = os.path.join(path_pannuke, 'fold_2')
path_fold3 = os.path.join(path_pannuke, 'fold_3')

path_mask_fold1 = os.path.join(path_fold1, 'masks/masks.npy')
path_mask_fold2 = os.path.join(path_fold2, 'masks/masks.npy')
path_mask_fold3 = os.path.join(path_fold3, 'masks/masks.npy')

path_image_fold1 = os.path.join(path_fold1, 'images/images.npy')
path_image_fold2 = os.path.join(path_fold2, 'images/images.npy')
path_image_fold3 = os.path.join(path_fold3, 'images/images.npy')

path_images = os.path.join(path_pannuke, 'images')

df_train = pd.read_csv(os.path.join(path_split,  'train_df.csv'), index_col = 0)
df_test = pd.read_csv(os.path.join(path_split, 'test_df.csv'), index_col = 0)
df_val = pd.read_csv(os.path.join(path_split, 'val_df.csv'), index_col = 0)



###################### Training parameters ###########################

parameters = {}
parameters['dim'] = 256
parameters['batch_size'] = 32
parameters['lr'] = 1e-4
