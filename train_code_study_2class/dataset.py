from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.transform import resize
from scipy import ndimage

from PIL import Image

import copy
import os
import torch
import six

import cv2

import albumentations as A
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


class COVIDDataset(datasets.VisionDataset):
    def __init__(self, root, df, mask=None, img_size=800,
                 transform=None, target_transform=None, crop_with_lung=False,
                 xy=True, mask_img_size=512, use_cache=True):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.img_size = img_size
        self.root = root
        self.xy = xy
        self.mask = mask
        self.mask_img_size = mask_img_size
        self.use_cache = use_cache
        self.crop_with_lung = crop_with_lung

        self.id_original_img_size_map = self.get_img_meta_map(root)

        df = df[[
            "id","boxes","label","StudyInstanceUID",
            "Negative for Pneumonia","Typical Appearance",
            "Indeterminate Appearance","Atypical Appearance",
        ]]
        self.data, img_ids = self.get_data(df)

        self.img_id_map = dict([(v,k) for k, v in enumerate(img_ids)])
        self.image_label_map = {0: "none", 1: "opacity"}
        self.study_label_map = {0: 'Negative for Pneumonia', 1: 'Typical Appearance', 2: 'Indeterminate Appearance', 3: 'Atypical Appearance'}
        self.image_label_map_inv = dict([(v, k) for k, v in self.image_label_map.items()])
        self.study_label_map_inv = dict([(v, k) for k, v in self.study_label_map.items()])

        self.loader = default_loader

    def __getitem__(self, index):
        img_id, study_id, label, boxes, study_targets = self.data[index]
        path = os.path.join(self.root, "train", f"{img_id.replace('_image','')}.png")

        img = self.loader(path)

        if self.mask:
            image_level_target, boxes, box_masks = self.parse_label(label, boxes, img_id, mask=self.mask)

            if self.mask in ["lung", "both", "mult"] or self.crop_with_lung:
                cached_dir = os.path.join(self.root, f"train_cache")
                cached_path = os.path.join(cached_dir, f"{img_id}_mask{self.mask}.png.npy")
                if os.path.isfile(cached_path) and self.use_cache:
                    masks_base = np.load(cached_path)
                else:
                    masks_base = copy.deepcopy(img)
                    if type(masks_base) == Image.Image:
                        masks_base = np.array(masks_base)

                    masks_base = cv2.cvtColor(masks_base, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    masks_base = clahe.apply(masks_base)
                    thresh = threshold_otsu(masks_base)
                    masks_base = 1 - (masks_base > thresh).astype(int)

                    masks_base = resize(resize(clear_border(masks_base)*255, (70, 70)), (self.img_size, self.img_size))
                    masks_base = clear_border((((masks_base / masks_base.max())*255)>30).astype("uint8"))
                    masks_base = self.remove_small_segments(masks_base)
                    masks_base = ndimage.binary_fill_holes(masks_base)
                    if self.mask == "both":
                        masks_base_0 = masks_base.astype(bool).astype(np.uint8)
                        masks_base_1 = box_masks.bool().numpy().astype(np.uint8)
                        masks_base = np.transpose(np.stack((masks_base_0, masks_base_1)), (1, 2, 0))
                    if self.mask == "mult":
                        masks_base = masks_base.astype(bool).astype(np.uint8) * box_masks.bool().numpy().astype(np.uint8)
                    os.makedirs(cached_dir, exist_ok=True)
                    with open(cached_path, 'wb') as f:
                        np.save(f, masks_base)
            else:
                masks_base = box_masks.bool().numpy()

            if self.mask == "both":
                img = np.concatenate((masks_base, np.array(img)), axis=2)
            else:
                img = np.concatenate((np.expand_dims(masks_base, 2), np.array(img)), axis=2)

            if self.crop_with_lung:
                if np.sum(masks_base) > 10:
                    assert self.mask != "both" and self.mask != "bbox"
                    area = np.argwhere(masks_base)

                    x1, y1 = np.min(area,axis=0)
                    x2, y2 = np.max(area,axis=0)

                    width = x2 - x1
                    height = y2 - y1
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    square_length = max(width, height)

                    x1 = max(0, int(x_center - square_length / 2))
                    x2 = min(self.img_size, int(x_center + square_length / 2))
                    y1 = max(0, int(y_center - square_length / 2))
                    y2 = min(self.img_size, int(y_center + square_length / 2))

                    # print(x1,x2,y1,y2)
                    # plt.imshow(img[:,:,1])
                    # plt.show()
                    img = img[x1:x2,y1:y2,:]
                    # plt.imshow(img[:,:,1])
                    # plt.show()
                else:
                    x_center = self.img_size // 2
                    y_center = self.img_size // 2

                    center_crop_size = int((self.img_size // 2) * 0.7)

                    x1 = x_center - center_crop_size - self.img_size // 9
                    x2 = x_center + center_crop_size - self.img_size // 9
                    y1 = y_center - center_crop_size
                    y2 = y_center + center_crop_size
                    img = img[x1:x2,y1:y2,:]
        else:
            image_level_target, boxes = self.parse_label(label, boxes, img_id)

        if self.transform is not None:
            if type(self.transform) == A.Compose:
                img = self.transform(image=np.array(img))["image"]
            else:
                img = self.transform(img)

        if self.mask:
            if img.shape[0] == 5:
                masks = img[0:2,:,:]
                masks = F.interpolate(masks.unsqueeze(0), size=self.mask_img_size).squeeze(0)

                img = img[2:,:,:]
            else:
                masks = F.interpolate(img[0,:,:].unsqueeze(0).unsqueeze(0), size=self.mask_img_size).squeeze(0)

                img = img[1:,:,:]

        study_level_target = torch.as_tensor(study_targets, dtype=torch.int64)

        if self.mask:
            if not self.xy:
                boxes[:,[0,1,2,3]] = boxes[:,[1,0,3,2]]
            return img, img_id, study_id, boxes, image_level_target, study_level_target, masks
        else:
            if not self.xy:
                boxes[:,[0,1,2,3]] = boxes[:,[1,0,3,2]]
            return img, study_level_target

    def __len__(self):
        return len(self.data)

    def get_img_meta_map(self, root):
        meta_df = pd.read_csv(os.path.join(root, "meta.csv"))
        meta_df = meta_df[meta_df["split"] == "train"]

        id_original_img_size_map = {}
        for row in meta_df.to_numpy():
            img_id, dim0, dim1, split = row
            id_original_img_size_map[img_id] = (dim0, dim1)

        return id_original_img_size_map

    def get_data(self, df):
        data = []
        img_ids = []
        for row in df.to_numpy():
            img_id, boxes, label, study_id, nfp, ta, ia, aa = row

            data.append((img_id, study_id, label, boxes, np.argmax([nfp, ta, ia, aa])))
            img_ids.append(img_id)
        return data, img_ids

    def parse_label(self, label, boxes, img_id, mask=None):
        splits = label.split(" ")
        num_labels = len(splits) // 6
        boxes_parsed = []
        for idx in range(num_labels):
            coordinates_abs = [float(splits[idx*6+2]), float(splits[idx*6+3]), float(splits[idx*6+4]), float(splits[idx*6+5])]

            if not pd.isna(boxes):
                box = eval(boxes)[idx]
                assert box["x"] + box["width"] == coordinates_abs[-2]
                assert box["y"] + box["height"] == coordinates_abs[-1]

                original_y, original_x = self.id_original_img_size_map[img_id.replace("_image", "")]

                coordinates_abs[0] = coordinates_abs[0] * (self.img_size / original_x)
                coordinates_abs[2] = coordinates_abs[2] * (self.img_size / original_x)

                coordinates_abs[1] = coordinates_abs[1] * (self.img_size / original_y)
                coordinates_abs[3] = coordinates_abs[3] * (self.img_size / original_y)

                boxes_parsed.append(coordinates_abs)

        if len(boxes_parsed) > 0:
            if mask:
                seg_mask = torch.sum(torch.stack(tuple(map(lambda x: self.get_mask(x), boxes_parsed))), dim=0)
                seg_mask = torch.clamp(seg_mask, max=1)
            boxes_parsed = torch.stack(tuple(map(torch.tensor, boxes_parsed)))
            empty = torch.as_tensor([1]*len(boxes_parsed), dtype=torch.int64)
        else:
            if mask:
                seg_mask = self.get_mask(None)
            boxes_parsed.append([0,0,1,1])
            boxes_parsed = torch.as_tensor(boxes_parsed, dtype=torch.float32)
            empty = torch.as_tensor([0], dtype=torch.int64)

        if mask:
            return empty, boxes_parsed, seg_mask
        else:
            return empty, boxes_parsed

    def get_mask(self, box):
        mask = torch.zeros((self.img_size, self.img_size))

        if box is None:
            return mask
        mask[int(box[1]):int(box[3]),int(box[0]):int(box[2])] = 1
        return mask

    def remove_small_segments(self, img):
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        min_size = 10000

        #your answer image
        new_img = np.zeros((output.shape))
        #for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                new_img[output == i + 1] = 255

        return new_img

class COVID2CDataset(datasets.VisionDataset):
    def __init__(self, root, df, mask=None, img_size=800,
                 transform=None, target_transform=None,
                 mask_img_size=512, use_cache=True):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.img_size = img_size
        self.root = root
        self.mask = mask
        self.mask_img_size = mask_img_size
        self.use_cache = use_cache

        self.id_original_img_size_map = self.get_img_meta_map(root)

        df = df[[
            "id","boxes","label","StudyInstanceUID","none"
        ]]
        self.data, img_ids = self.get_data(df)

        self.img_id_map = dict([(v,k) for k, v in enumerate(img_ids)])
        self.image_label_map = {0: "none", 1: "opacity"}
        self.study_label_map = {0: 'Negative for Pneumonia', 1: 'Typical Appearance', 2: 'Indeterminate Appearance', 3: 'Atypical Appearance'}
        self.image_label_map_inv = dict([(v, k) for k, v in self.image_label_map.items()])
        self.study_label_map_inv = dict([(v, k) for k, v in self.study_label_map.items()])

        self.loader = default_loader

    def __getitem__(self, index):
        img_id, study_id, label, boxes, study_targets = self.data[index]
        path = os.path.join(self.root, "train", f"{img_id.replace('_image','')}.png")

        img = self.loader(path)

        if self.mask:
            image_level_target, boxes, box_masks = self.parse_label(label, boxes, img_id, mask=self.mask)

            if self.mask in ["lung", "both"]:
                cached_dir = os.path.join(self.root, f"train_cache_{self.mask_img_size}")
                cached_path = os.path.join(cached_dir, f"{img_id}_mask{self.mask}.png.npy")
                if os.path.isfile(cached_path) and self.use_cache:
                    masks_base = np.load(cached_path)
                else:
                    masks_base = copy.deepcopy(img)
                    if type(masks_base) == Image.Image:
                        masks_base = np.array(masks_base)

                    masks_base = cv2.cvtColor(masks_base, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    masks_base = clahe.apply(masks_base)
                    thresh = threshold_otsu(masks_base)
                    masks_base = 1 - (masks_base > thresh).astype(int)

                    masks_base = resize(resize(clear_border(masks_base)*255, (70, 70)), (self.img_size, self.img_size))
                    masks_base = clear_border((((masks_base / masks_base.max())*255)>30).astype("uint8"))
                    masks_base = self.remove_small_segments(masks_base)
                    masks_base = ndimage.binary_fill_holes(masks_base)
                    if self.mask == "both":
                        masks_base_0 = masks_base.astype(bool).astype(np.uint8)
                        masks_base_1 = box_masks.bool().numpy().astype(np.uint8)
                        masks_base = np.transpose(np.stack((masks_base_0, masks_base_1)), (1, 2, 0))
                    os.makedirs(cached_dir, exist_ok=True)
                    with open(cached_path, 'wb') as f:
                        np.save(f, masks_base)
            else:
                masks_base = box_masks.bool().numpy()

            if self.mask == "both":
                img = np.concatenate((masks_base, np.array(img)), axis=2)
            else:
                img = np.concatenate((np.expand_dims(masks_base, 2), np.array(img)), axis=2)
        else:
            image_level_target, boxes = self.parse_label(label, boxes, img_id)

        if self.transform is not None:
            if type(self.transform) == A.Compose:
                img = self.transform(image=np.array(img))["image"]
            else:
                img = self.transform(img)

        if self.mask:
            if img.shape[0] == 5:
                masks = img[0:2,:,:]
                masks = F.interpolate(masks.unsqueeze(0), size=self.mask_img_size).squeeze(0)

                img = img[2:,:,:]
            else:
                masks = img[0,:,:].unsqueeze(0)
                img = img[1:,:,:]

        study_level_target = torch.as_tensor(study_targets, dtype=torch.int64)

        if self.mask:
            return img, img_id, study_id, None, None, study_level_target, masks
        else:
            return img, study_level_target

    def __len__(self):
        return len(self.data)

    def get_img_meta_map(self, root):
        meta_df = pd.read_csv(os.path.join(root, "meta.csv"))
        meta_df = meta_df[meta_df["split"] == "train"]

        id_original_img_size_map = {}
        for row in meta_df.to_numpy():
            img_id, dim0, dim1, split = row
            id_original_img_size_map[img_id] = (dim0, dim1)

        return id_original_img_size_map

    def get_data(self, df):
        data = []
        img_ids = []
        for row in df.to_numpy():
            img_id, boxes, label, study_id, none = row

            data.append((img_id, study_id, label, boxes, none))
            img_ids.append(img_id)
        return data, img_ids

    def parse_label(self, label, boxes, img_id, mask=None):
        splits = label.split(" ")
        num_labels = len(splits) // 6
        boxes_parsed = []
        for idx in range(num_labels):
            coordinates_abs = [float(splits[idx*6+2]), float(splits[idx*6+3]), float(splits[idx*6+4]), float(splits[idx*6+5])]

            if not pd.isna(boxes):
                box = eval(boxes)[idx]
                assert box["x"] + box["width"] == coordinates_abs[-2]
                assert box["y"] + box["height"] == coordinates_abs[-1]

                original_y, original_x = self.id_original_img_size_map[img_id.replace("_image", "")]

                coordinates_abs[0] = coordinates_abs[0] * (self.img_size / original_x)
                coordinates_abs[2] = coordinates_abs[2] * (self.img_size / original_x)

                coordinates_abs[1] = coordinates_abs[1] * (self.img_size / original_y)
                coordinates_abs[3] = coordinates_abs[3] * (self.img_size / original_y)

                boxes_parsed.append(coordinates_abs)

        if len(boxes_parsed) > 0:
            if mask:
                seg_mask = torch.sum(torch.stack(tuple(map(lambda x: self.get_mask(x), boxes_parsed))), dim=0)
                seg_mask = torch.clamp(seg_mask, max=1)
            boxes_parsed = torch.stack(tuple(map(torch.tensor, boxes_parsed)))
            empty = torch.as_tensor([1]*len(boxes_parsed), dtype=torch.int64)
        else:
            if mask:
                seg_mask = self.get_mask(None)
            boxes_parsed.append([0,0,1,1])
            boxes_parsed = torch.as_tensor(boxes_parsed, dtype=torch.float32)
            empty = torch.as_tensor([0], dtype=torch.int64)

        if mask:
            return empty, boxes_parsed, seg_mask
        else:
            return empty, boxes_parsed

    def get_mask(self, box):
        mask = torch.zeros((self.img_size, self.img_size))

        if box is None:
            return mask
        mask[int(box[1]):int(box[3]),int(box[0]):int(box[2])] = 1
        return mask

    def remove_small_segments(self, img):
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        min_size = 10000

        #your answer image
        new_img = np.zeros((output.shape))
        #for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                new_img[output == i + 1] = 255

        return new_img


class DistillCOVIDDataset(datasets.VisionDataset):
    def __init__(self, root, df, transform=None, target_transform=None, img_size=800, crop_with_lung=False, use_cache=True):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.img_size = img_size
        self.root = root
        self.id_original_img_size_map = self.get_img_meta_map(root)
        self.crop_with_lung = crop_with_lung
        self.use_cache = use_cache

        df = df[[
            "id","StudyInstanceUID",
            "Negative for Pneumonia","Typical Appearance",
            "Indeterminate Appearance","Atypical Appearance",
            "psl_Negative for Pneumonia","psl_Typical Appearance",
            "psl_Indeterminate Appearance","psl_Atypical Appearance",
        ]]
        self.data, img_ids = self.get_data(df)

        self.img_id_map = dict([(v,k) for k, v in enumerate(img_ids)])
        self.image_label_map = {0: "none", 1: "opacity"}
        self.study_label_map = {0: 'Negative for Pneumonia', 1: 'Typical Appearance', 2: 'Indeterminate Appearance', 3: 'Atypical Appearance'}
        self.image_label_map_inv = dict([(v, k) for k, v in self.image_label_map.items()])
        self.study_label_map_inv = dict([(v, k) for k, v in self.study_label_map.items()])

        self.loader = default_loader

    def __getitem__(self, index):
        image_id, study_id, targets, psl_targets = self.data[index]
        path = os.path.join(self.root, "train", f"{image_id.replace('_image','')}.png")

        img = self.loader(path)
        img = np.array(img)

        if self.crop_with_lung:
            cached_dir = os.path.join(self.root, f"train_cache_384")
            cached_path = os.path.join(cached_dir, f"{image_id}_masklung.png.npy")
            if os.path.isfile(cached_path) and self.use_cache:
                masks_base = np.load(cached_path)
            else:
                masks_base = copy.deepcopy(img)
                if type(masks_base) == Image.Image:
                    masks_base = np.array(masks_base)

                masks_base = cv2.cvtColor(masks_base, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                masks_base = clahe.apply(masks_base)
                thresh = threshold_otsu(masks_base)
                masks_base = 1 - (masks_base > thresh).astype(int)

                masks_base = resize(resize(clear_border(masks_base)*255, (70, 70)), (self.img_size, self.img_size))
                masks_base = clear_border((((masks_base / masks_base.max())*255)>30).astype("uint8"))
                masks_base = self.remove_small_segments(masks_base)
                masks_base = ndimage.binary_fill_holes(masks_base)
                os.makedirs(cached_dir, exist_ok=True)
                with open(cached_path, 'wb') as f:
                    np.save(f, masks_base)

            if np.sum(masks_base) > 10:
                area = np.argwhere(masks_base)

                x1, y1 = np.min(area,axis=0)
                x2, y2 = np.max(area,axis=0)

                width = x2 - x1
                height = y2 - y1
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                square_length = max(width, height)

                x1 = max(0, int(x_center - square_length / 2))
                x2 = min(self.img_size, int(x_center + square_length / 2))
                y1 = max(0, int(y_center - square_length / 2))
                y2 = min(self.img_size, int(y_center + square_length / 2))

                img = img[x1:x2,y1:y2,:]
            else:
                x_center = self.img_size // 2
                y_center = self.img_size // 2

                center_crop_size = int((self.img_size // 2) * 0.7)

                x1 = x_center - center_crop_size - self.img_size // 9
                x2 = x_center + center_crop_size - self.img_size // 9
                y1 = y_center - center_crop_size
                y2 = y_center + center_crop_size
                img = img[x1:x2,y1:y2,:]

        if self.transform is not None:
            if type(self.transform) == A.Compose:
                img = self.transform(image=img)["image"]
            else:
                img = self.transform(img)

        targets = torch.as_tensor(targets, dtype=torch.float32)
        psl_targets = torch.as_tensor(psl_targets, dtype=torch.float32)

        return img.float(), targets, psl_targets

    def __len__(self):
        return len(self.data)

    def get_img_meta_map(self, root):
        meta_df = pd.read_csv(os.path.join(root, "meta.csv"))
        meta_df = meta_df[meta_df["split"] == "train"]

        id_original_img_size_map = {}
        for row in meta_df.to_numpy():
            img_id, dim0, dim1, split = row
            id_original_img_size_map[img_id] = (dim0, dim1)

        return id_original_img_size_map

    def get_data(self, df):
        data = []
        img_ids = []
        for row in df.to_numpy():
            image_id, study_id, nfp, ta, ia, aa, nfp_psl, ta_psl, ia_psl, aa_psl = row
            data.append((image_id, study_id, [nfp, ta, ia, aa], [nfp_psl, ta_psl, ia_psl, aa_psl]))
            img_ids.append(image_id)
        return data, img_ids

    def remove_small_segments(self, img):
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        min_size = 10000

        #your answer image
        new_img = np.zeros((output.shape))
        #for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                new_img[output == i + 1] = 255

        return

class VinBigDataset(datasets.VisionDataset):
    def __init__(self, root, df, mask=None, img_size=800,
                 transform=None, target_transform=None, crop_with_lung=False,
                 xy=True, mask_img_size=512, use_cache=True):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.img_size = img_size
        self.root = root
        self.xy = xy
        self.mask = mask
        self.mask_img_size = mask_img_size
        self.use_cache = use_cache
        self.crop_with_lung = crop_with_lung

        self.id_original_img_size_map = self.get_img_meta_map(root)

        df = df[[
            "id","boxes","label"
        ]]
        self.data, img_ids = self.get_data(df)

        self.loader = default_loader

    def __getitem__(self, index):
        img_id, study_id, label, boxes, study_targets = self.data[index]
        path = os.path.join(self.root, "train", f"{img_id.replace('_image','')}.png")

        img = self.loader(path)

        if self.mask:
            image_level_target, boxes, box_masks = self.parse_label(label, boxes, img_id, mask=self.mask)

            if self.mask in ["lung", "both", "mult"] or self.crop_with_lung:
                cached_dir = os.path.join(self.root, f"train_cache_{self.mask_img_size}")
                cached_path = os.path.join(cached_dir, f"{img_id}_mask{self.mask}.png.npy")
                if os.path.isfile(cached_path) and self.use_cache:
                    masks_base = np.load(cached_path)
                else:
                    masks_base = copy.deepcopy(img)
                    if type(masks_base) == Image.Image:
                        masks_base = np.array(masks_base)

                    masks_base = cv2.cvtColor(masks_base, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    masks_base = clahe.apply(masks_base)
                    thresh = threshold_otsu(masks_base)
                    masks_base = 1 - (masks_base > thresh).astype(int)

                    masks_base = resize(resize(clear_border(masks_base)*255, (70, 70)), (self.img_size, self.img_size))
                    masks_base = clear_border((((masks_base / masks_base.max())*255)>30).astype("uint8"))
                    masks_base = self.remove_small_segments(masks_base)
                    masks_base = ndimage.binary_fill_holes(masks_base)
                    if self.mask == "both":
                        masks_base_0 = masks_base.astype(bool).astype(np.uint8)
                        masks_base_1 = box_masks.bool().numpy().astype(np.uint8)
                        masks_base = np.transpose(np.stack((masks_base_0, masks_base_1)), (1, 2, 0))
                    if self.mask == "mult":
                        masks_base = masks_base.astype(bool).astype(np.uint8) * box_masks.bool().numpy().astype(np.uint8)
                    os.makedirs(cached_dir, exist_ok=True)
                    with open(cached_path, 'wb') as f:
                        np.save(f, masks_base)
            else:
                masks_base = box_masks.bool().numpy()

            if self.mask == "both":
                img = img.resize((640, 640))
                img = np.concatenate((masks_base, np.array(img)), axis=2)
            else:
                img = np.concatenate((np.expand_dims(masks_base, 2), np.array(img)), axis=2)

            if self.crop_with_lung:
                if np.sum(masks_base) > 10:
                    assert self.mask != "both" and self.mask != "bbox"
                    area = np.argwhere(masks_base)

                    x1, y1 = np.min(area,axis=0)
                    x2, y2 = np.max(area,axis=0)

                    width = x2 - x1
                    height = y2 - y1
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    square_length = max(width, height)

                    x1 = max(0, int(x_center - square_length / 2))
                    x2 = min(self.img_size, int(x_center + square_length / 2))
                    y1 = max(0, int(y_center - square_length / 2))
                    y2 = min(self.img_size, int(y_center + square_length / 2))

                    # print(x1,x2,y1,y2)
                    # plt.imshow(img[:,:,1])
                    # plt.show()
                    img = img[x1:x2,y1:y2,:]
                    # plt.imshow(img[:,:,1])
                    # plt.show()
                else:
                    x_center = self.img_size // 2
                    y_center = self.img_size // 2

                    center_crop_size = int((self.img_size // 2) * 0.7)

                    x1 = x_center - center_crop_size - self.img_size // 9
                    x2 = x_center + center_crop_size - self.img_size // 9
                    y1 = y_center - center_crop_size
                    y2 = y_center + center_crop_size
                    img = img[x1:x2,y1:y2,:]
        else:
            image_level_target, boxes = self.parse_label(label, boxes, img_id)

        if self.transform is not None:
            if type(self.transform) == A.Compose:
                img = self.transform(image=np.array(img))["image"]
            else:
                img = self.transform(img)

        if self.mask:
            if img.shape[0] == 5:
                masks = img[0:2,:,:]
                masks = F.interpolate(masks.unsqueeze(0), size=self.mask_img_size).squeeze(0)

                img = img[2:,:,:]
            else:
                masks = F.interpolate(img[0,:,:].unsqueeze(0).unsqueeze(0), size=self.mask_img_size).squeeze(0)

                img = img[1:,:,:]

        # study_level_target = torch.as_tensor(study_targets, dtype=torch.int64)

        if self.mask:
            if not self.xy:
                boxes[:,[0,1,2,3]] = boxes[:,[1,0,3,2]]
            return img, img_id, study_id, boxes, image_level_target, torch.ones(1), masks
        else:
            if not self.xy:
                boxes[:,[0,1,2,3]] = boxes[:,[1,0,3,2]]
            return img, img_id, study_id, boxes, image_level_target, torch.ones(1)

    def __len__(self):
        return len(self.data)

    def get_img_meta_map(self, root):
        meta_df = pd.read_csv(os.path.join(root, "meta.csv"))
        # meta_df = meta_df[meta_df["split"] == "train"]

        id_original_img_size_map = {}
        for row in meta_df.to_numpy():
            img_id, dim0, dim1 = row
            id_original_img_size_map[img_id] = (dim0, dim1)

        return id_original_img_size_map

    def get_data(self, df):
        data = []
        img_ids = []
        for row in df.to_numpy():
            img_id, boxes, label = row

            data.append((img_id, None, label, boxes, None))
            img_ids.append(img_id)
        return data, img_ids

    def parse_label(self, label, boxes, img_id, mask=None):
        splits = label.split(" ")
        num_labels = len(splits) // 6
        boxes_parsed = []
        for idx in range(num_labels):
            coordinates_abs = [float(splits[idx*6+2]), float(splits[idx*6+3]), float(splits[idx*6+4]), float(splits[idx*6+5])]

            if not pd.isna(boxes):
                box = eval(boxes)[idx]
                assert box["x"] + box["width"] == coordinates_abs[-2]
                assert box["y"] + box["height"] == coordinates_abs[-1]

                original_y, original_x = self.id_original_img_size_map[img_id.replace("_image", "")]

                coordinates_abs[0] = coordinates_abs[0] * (self.img_size / original_x)
                coordinates_abs[2] = coordinates_abs[2] * (self.img_size / original_x)

                coordinates_abs[1] = coordinates_abs[1] * (self.img_size / original_y)
                coordinates_abs[3] = coordinates_abs[3] * (self.img_size / original_y)

                boxes_parsed.append(coordinates_abs)

        if len(boxes_parsed) > 0:
            if mask:
                seg_mask = torch.sum(torch.stack(tuple(map(lambda x: self.get_mask(x), boxes_parsed))), dim=0)
                seg_mask = torch.clamp(seg_mask, max=1)
            boxes_parsed = torch.stack(tuple(map(torch.tensor, boxes_parsed)))
            empty = torch.as_tensor([1]*len(boxes_parsed), dtype=torch.int64)
        else:
            if mask:
                seg_mask = self.get_mask(None)
            boxes_parsed.append([0,0,1,1])
            boxes_parsed = torch.as_tensor(boxes_parsed, dtype=torch.float32)
            empty = torch.as_tensor([0], dtype=torch.int64)

        if mask:
            return empty, boxes_parsed, seg_mask
        else:
            return empty, boxes_parsed

    def get_mask(self, box):
        mask = torch.zeros((self.img_size, self.img_size))

        if box is None:
            return mask
        mask[int(box[1]):int(box[3]),int(box[0]):int(box[2])] = 1
        return mask

    def remove_small_segments(self, img):
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        min_size = 10000

        #your answer image
        new_img = np.zeros((output.shape))
        #for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                new_img[output == i + 1] = 255

        return new_img


class NIHDataset(datasets.VisionDataset):
    def __init__(self, root, df, transform=None, target_transform=None, img_size=800):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.img_size = img_size
        self.root = root

        self.label_map = {
            'Nodule': 0,
            'Fibrosis': 1,
            'Cardiomegaly': 2,
            'Effusion': 3,
            'Pleural_Thickening': 4,
            'No Finding': 5,
            'Mass': 6,
            'Atelectasis': 7,
            'Edema': 8,
            'Consolidation': 9,
            'Infiltration': 10,
            'Pneumonia': 11,
            'Emphysema': 12,
            'Pneumothorax': 13,
            'Hernia': 14
        }

        df = df[["id", "Finding Labels"]]
        self.data = self.get_data(df)

        self.loader = default_loader

    def __getitem__(self, index):
        image_id, targets = self.data[index]
        path = os.path.join(self.root, "train", image_id)

        try:
            img = self.loader(path)
            img = np.array(img)
        except OSError:
            print(path)
            return None

        if self.transform is not None:
            if type(self.transform) == A.Compose:
                img = self.transform(image=img)["image"]
            else:
                img = self.transform(img)

        targets = torch.nn.functional.one_hot(torch.as_tensor(targets, dtype=torch.long), num_classes=15)

        return img.float(), torch.sum(targets.float(), dim=0)

    def get_data(self, df):
        data = []
        for row in df.to_numpy():
            image_id, label = row
            label = [self.label_map[i] for i in label.split("|")]
            data.append((image_id, label))
        return data

    def __len__(self):
        return len(self.data)