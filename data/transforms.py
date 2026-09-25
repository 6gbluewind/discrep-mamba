import os
import cc3d
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import random
from torch import (
    Tensor,
    load as torch_load,
    from_numpy as torch_from_numpy,
    cat as torch_cat,
    logical_or as torch_logical_or,
    logical_and as torch_logical_and,
    where as torch_where,
)
from typing import List, Union, Sequence
from monai.data import MetaTensor
from monai.config import KeysCollection
from monai.transforms import *
from omegaconf import ListConfig
from copy import deepcopy


class CropToMask(MapTransform):
    """
    Crop the images to a region-of-interest bounding box defined by the segmentation masks.

        keys: datadict key for images that the transform should be applied to
        mask_key: datadict key for the segmentation masks to be used as a look-up
        use_class: which class in the mask to be cropped to
        margin: optional margin to extend the bounding box in each direction
    """

    def __init__(
        self,
        keys: KeysCollection,
        mask_key: str,
        use_class: Union[int, List[int]] = 1,
        margin: int = 0,
        min_dim: tuple[int, int, int] = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        assert mask_key is not None
        self.mask_key = mask_key
        self.margin = margin
        self.min_dim = min_dim

        if isinstance(use_class, int):
            self.use_class = [use_class for _ in range(len(self.keys))]
        elif isinstance(use_class, list):
            assert len(use_class) == len(
                self.keys
            ), "If provided separately for each key, use_class list needs to have same length as keys."
            self.use_class = use_class

    def __call__(self, data):
        # only load the mask image for the first time, then store bbox info in datadict for next epochs
        for key, use_class in zip(self.keys, self.use_class):
            if f"crop_bounding_box:{use_class}" not in data:
                mask = data[self.mask_key].squeeze(0).numpy()
                try:
                    (
                        data[f"crop_bounding_box:{use_class}"],
                        data[f"pad_bounding_box:{use_class}"],
                    ) = CropToMask.get_bbox(mask, self.margin, self.min_dim, use_class)
                except Exception as e:
                    print(f"Error in patient {data['patient_id']}: {e}")
                    print(f"{self.use_class=}")
                    raise e
            bbox = data[f"crop_bounding_box:{use_class}"]
            bbox_start = Tensor([i[0] for i in bbox])
            padding = [
                i
                for tuples in data[f"pad_bounding_box:{use_class}"]
                for i in list(reversed(tuples))
            ]
            if key in data:
                data[key] = data[key][
                    :,
                    int(bbox[0][0]) : int(bbox[0][1]) + 1,
                    int(bbox[1][0]) : int(bbox[1][1]) + 1,
                    int(bbox[2][0]) : int(bbox[2][1]) + 1,
                ]

                # this works since padding must be a multiple of 2
                # padding after reverse:
                # [d_start, d_end, w_start, w_end, h_start, h_end]
                # padding before reverse:
                # [h_end, h_start, w_end, w_start, d_end, d_start]
                # -> need h_start, w_start and d_start to calculate crop offset
                # thus the uneven indices
                padding_offset = Tensor(
                    [-padding[2 * i + 1] for i in range(len(padding) // 2)]
                ).long()

                padding.reverse()
                data[key] = F.pad(data[key], tuple(padding), "constant", 0)
                print(f"{bbox_start.shape=}")
                crop_end = bbox_start + Tensor(data[key].shape[1:])
                crop_start = bbox_start + padding_offset

                if f"{key}_crop_start" in data:
                    prev_crop_start = data[f"{key}_crop_start"]
                    data[f"{key}_crop_end"] = prev_crop_start + crop_end
                    data[f"{key}_crop_start"] = prev_crop_start + crop_start
                else:
                    data[f"{key}_crop_start"] = crop_start
                    data[f"{key}_crop_end"] = crop_end
                print(f"{data[f'{key}_crop_start']=}")
                print(f"{data[f'{key}_crop_end']=}")
            elif not self.allow_missing_keys:
                raise KeyError(f"Specified key '{key}' not found in data.")

        return data

    @staticmethod
    def get_bbox(
        mask: np.array,
        margin: int = 0,
        min_dim: tuple[int, int, int] = None,
        use_class: int = 1,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:

        img = mask.copy()
        img = np.where(img == use_class, 1, 0)  # take only the class of interest
        xmin, xmax = np.split(np.where(np.any(img, axis=(1, 2)))[0][[0, -1]], 2)
        ymin, ymax = np.split(np.where(np.any(img, axis=(0, 2)))[0][[0, -1]], 2)
        zmin, zmax = np.split(np.where(np.any(img, axis=(0, 1)))[0][[0, -1]], 2)
        bbox = ((xmin, xmax), (ymin, ymax), (zmin, zmax))

        new_bbox = []
        to_pad = []
        for i in range(3):
            minv, maxv = (
                bbox[i][0],
                bbox[i][1],
            )
            current_size = maxv - minv + 1
            # center = int(np.rint(np.mean([minv, maxv])))
            if not min_dim:
                cmargin = 0
            else:
                cmargin = (
                    min_dim[i] - current_size
                ) / 2  # int(np.floor(min_dim[i] / 2))
            minv = int(np.floor(minv - cmargin - margin))
            upper_add = abs(
                min(minv, 0)
            )  # if we hit the edge, don't pad and add to the other side
            maxv = int(np.floor(maxv + cmargin + margin + upper_add))
            lower_add = max(
                maxv - img.shape[i] + 1, 0
            )  # if we hit the edge, don't pad and add to the other side
            maxv = min(maxv, img.shape[i])
            minv = max(minv - lower_add, 0)
            new_bbox.append((minv, maxv))
            # if desired size is smaller than current size, pad equally on both sides
            new_size = maxv - minv + 1
            pad_min = max(int(np.floor((min_dim[i] - new_size) / 2)), 0)
            pad_max = max(int(np.ceil((min_dim[i] - new_size) / 2)), 0)
            to_pad.append((pad_min, pad_max))
        bbox = tuple(new_bbox)  # type: ignore

        return bbox, to_pad


class ExtractLargestTumor(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        mask_key: str,
        use_class: Union[int, List[int]] = 2,
        margin: int = 0,
        min_dim: tuple[int, int, int] = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        assert mask_key is not None
        self.mask_key = mask_key
        self.use_class = use_class

    def __call__(self, data):
        mask = data[self.mask_key].squeeze(0).numpy()
        edema = np.where(mask.copy() > 0, 1, 0)  # merge tumor and edema
        tumor = np.where(mask.copy() == self.use_class, 1, 0)  # take tumor only
        # perform CCA to separate distinct tumor, take largest one and then find the edema for it
        # largest edema does not always (in 6 patients) contain largest metastasis
        largest_metastasis = cc3d.largest_k(tumor, k=1)
        edemas = cc3d.connected_components(edema)

        edema_index = find_largest_intersection(largest_metastasis, edemas)
        mask = np.where(
            edemas.copy() == edema_index, mask, 0
        )  # select edema and metastasis that belongs to the largest metastasis
        data[self.mask_key] = torch_from_numpy(mask).unsqueeze(0)

        return data


def find_largest_intersection(largest_tumor, edemas):
    volumes = [0]
    for _, image in cc3d.each(edemas, binary=True, in_place=True):
        volumes.append((largest_tumor * image).sum())
    return np.asarray(volumes).argmax()


class CombineSegmentation(MapTransform):
    def __init__(self, keys, a_key, b_key, result_key, b_key_shift=100):
        super().__init__(keys)
        self.a_key = a_key
        self.b_key = b_key
        self.result_key = result_key
        self.b_key_shift = b_key_shift

    def __call__(self, data):
        a = data[self.a_key]
        b = data[self.b_key]
        b[b != 0] += self.b_key_shift
        b[b == 0] += a[b == 0]
        data[self.result_key] = b
        return data


# class ConvertBraTSClassesToMultiLabelsd(MapTransform):
#     """
#     Convert labels to multi channels based on brats classes:
#     label 1 is the non-enhancing tumor core
#     label 2 is the peritumoral edema
#     label 3 is the GD-enhancing tumor
#     The possible classes are TC (Tumor core), WT (Whole tumor)
#     and ET (Enhancing tumor).
#     """

#     def __init__(
#         self,
#         keys: KeysCollection,
#         allow_missing_keys: bool = False,
#         label_offset: int = 0,
#     ):
#         super().__init__(keys, allow_missing_keys)
#         self.label_offset = label_offset

#     def __call__(self, data):
#         d = dict(data)
#         for key in self.keys:
#             if key in d:
#                 result = []
#                 # merge label 1 and label 3 to construct TC
#                 result.append(
#                     torch_logical_or(
#                         d[key] == self.label_offset + 1, d[key] == self.label_offset + 3
#                     )
#                 )
#                 # merge labels 1, 2 and 3 to construct WT
#                 result.append(
#                     torch_logical_or(
#                         torch_logical_or(
#                             d[key] == self.label_offset + 2,
#                             d[key] == self.label_offset + 3,
#                         ),
#                         d[key] == self.label_offset + 1,
#                     )
#                 )
#                 # label 3 is ET
#                 result.append(d[key] == self.label_offset + 3)
#                 d[key] = torch_cat(result, axis=0)
#             elif not self.allow_missing_keys:
#                 raise KeyError(f"Specified key '{key}' not found in data.")

#         return d


# class MultiLabelsToBraTSClasses(Transform):
#     """
#     Convert multi-channel predictions to single channel labels
#     """

#     def __init__(self, label_offset: int = 0):
#         self.label_offset = label_offset

#     def __call__(self, outputs: Tensor):
#         # ET is label 3
#         single_channel = torch_where(
#             outputs[:, self.label_offset + 2] == 1, self.label_offset + 3, 0
#         )
#         # TC is label 3 and 1 merged -> exclude ET from TC to find 1
#         single_channel[
#             torch_logical_and(
#                 ~(outputs[:, self.label_offset + 2] == 1),
#                 outputs[:, self.label_offset + 0] == 1,
#             )
#         ] = (
#             self.label_offset + 1
#         )
#         # WT is all labels combined -> extract TC from WT
#         single_channel[
#             torch_logical_and(
#                 ~(outputs[:, self.label_offset + 0] == 1),
#                 outputs[:, self.label_offset + 1] == 1,
#             )
#         ] = (
#             self.label_offset + 2
#         )

#         return single_channel


class AddCropPositiond(MapTransform):
    def __init__(self, keys, target_size, original_size=None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        target_size = Tensor(target_size)
        self.keys = keys
        if original_size is not None:
            original_size = Tensor(original_size)
            self.delta_size = target_size - original_size
            self.crop_start = self.delta_size // 2
            self.crop_end = self.crop_start + original_size
        self.original_size = original_size
        self.target_size = target_size

    def __call__(self, data):
        if self.original_size is None:
            original_size = data[self.keys[0]].shape
            original_size = Tensor(list(original_size)).int()
            original_size = original_size[-3:]
            delta_size = self.target_size - original_size
            crop_start = delta_size // 2
            crop_end = crop_start + original_size
            self.original_size = original_size
            self.crop_start = crop_start
            self.crop_end = crop_end
        else:
            crop_start = self.crop_start
            crop_end = self.crop_end
        for key in self.keys:
            if key in data:
                if f"{key}_crop_start" in data:
                    print("already in data")
                data[f"{key}_crop_start"] = crop_start
                data[f"{key}_crop_end"] = crop_end
            else:
                if not self.allow_missing_keys:
                    raise KeyError(f"Specified key '{key}' not found in data.")
        data["crop_start"] = crop_start
        data["crop_end"] = crop_end
        return data


class CropForegroundd(CropForegroundd):
    def __call__(self, data, lazy: bool | None = None):
        d = dict(data)
        self.cropper: CropForeground
        box_start, box_end = self.cropper.compute_bounding_box(img=d[self.source_key])
        if self.start_coord_key is not None:
            d[self.start_coord_key] = box_start  # type: ignore
        if self.end_coord_key is not None:
            d[self.end_coord_key] = box_end  # type: ignore

        lazy_ = self.lazy if lazy is None else lazy
        for key, m in self.key_iterator(d, self.mode):
            d[key] = self.cropper.crop_pad(
                img=d[key], box_start=box_start, box_end=box_end, mode=m, lazy=lazy_
            )
            if (
                f"{key}_crop_start" in data
            ):  # if we stack random cropping after a center or foreground crop, we need to adjust the crop_start and crop_end
                prev_crop_start = data[f"{key}_crop_start"]
                data[f"{key}_crop_end"] = prev_crop_start + Tensor(box_end)
                data[f"{key}_crop_start"] = prev_crop_start + Tensor(box_start)
            else:
                d[f"{key}_crop_start"] = Tensor(box_start)
                d[f"{key}_crop_end"] = Tensor(box_end)
        return d


class CenterSpatialCropd(CenterSpatialCropd):
    def __call__(self, data, lazy: bool | None = None):
        # iterate, maybe different keys have different centres and thus different start/end coords
        for key in self.key_iterator(data):
            slices = self.cropper.compute_slices(data[key].shape[1:])
            if (
                f"{key}_crop_start" in data
            ):  # if we stack random cropping after a center or foreground crop, we need to adjust the crop_start and crop_end
                prev_crop_start = data[f"{key}_crop_start"]
                data[f"{key}_crop_end"] = prev_crop_start + Tensor(
                    [s.stop for s in slices]
                )
                data[f"{key}_crop_start"] = prev_crop_start + Tensor(
                    [s.start for s in slices]
                )
            else:
                data[f"{key}_crop_start"] = Tensor([s.start for s in slices])
                data[f"{key}_crop_end"] = Tensor([s.stop for s in slices])
        data = super().__call__(data, lazy=lazy)
        return data


class ResizeWithPadOrCropd(ResizeWithPadOrCropd):
    def __call__(self, data, lazy: bool | None = None):
        # iterate, maybe different keys have different centres and thus different start/end coords
        for key in self.key_iterator(data):
            slices = self.padder.cropper.compute_slices(data[key].shape[1:])

            if (
                f"{key}_crop_start" in data
            ):  # if we stack random cropping after a center or foreground crop, we need to adjust the crop_start and crop_end
                slice_start = Tensor([s.start for s in slices])
                slice_stop = Tensor([s.stop for s in slices])
                slice_size = slice_stop - slice_start

                crop_size = data[f"{key}_crop_end"] - data[f"{key}_crop_start"]
                delta_size = slice_size - crop_size

                start_change = delta_size // 2
                stop_change = delta_size - (delta_size // 2)

                new_start = data[f"{key}_crop_start"] - start_change
                new_stop = new_start + slice_size

                data[f"{key}_crop_end"] = new_stop
                data[f"{key}_crop_start"] = new_start
            else:
                data[f"{key}_crop_start"] = Tensor([s.start for s in slices])
                data[f"{key}_crop_end"] = Tensor([s.stop for s in slices])
        data = super().__call__(data, lazy=lazy)
        return data


class SpatialCropd(SpatialCropd):
    def __init__(
        self,
        keys: KeysCollection,
        roi_center: Sequence[int] | int | None = None,
        roi_size: Sequence[int] | int | None = None,
        roi_start: Sequence[int] | int | None = None,
        roi_end: Sequence[int] | int | None = None,
        roi_slices: Sequence[slice] | None = None,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        if roi_center is not None and isinstance(roi_center, ListConfig):
            roi_center = list(roi_center)
        if roi_size is not None and isinstance(roi_size, ListConfig):
            roi_size = list(roi_size)
        if roi_start is not None and isinstance(roi_start, ListConfig):
            roi_start = list(roi_start)
        if roi_end is not None and isinstance(roi_end, ListConfig):
            roi_end = list(roi_end)
        if roi_slices is not None and isinstance(roi_slices, ListConfig):
            roi_slices = list(roi_slices)
        super().__init__(
            keys,
            roi_center=roi_center,
            roi_size=roi_size,
            roi_start=roi_start,
            roi_end=roi_end,
            roi_slices=roi_slices,
            allow_missing_keys=allow_missing_keys,
            lazy=lazy,
        )

    def __call__(self, data, lazy: bool | None = None):
        data = super().__call__(data, lazy=lazy)
        # no need to iterate, all keys will be cropped the same way
        slices = self.cropper.slices
        crop_start = Tensor([s.start for s in slices])
        crop_end = Tensor([s.stop for s in slices])
        for key in self.key_iterator(data):
            if (
                f"{key}_crop_start" in data
            ):  # if we stack random cropping after a center or foreground crop, we need to adjust the crop_start and crop_end
                prev_crop_start = data[f"{key}_crop_start"]
                data[f"{key}_crop_end"] = prev_crop_start + crop_end
                data[f"{key}_crop_start"] = prev_crop_start + crop_start
            else:
                data[f"{key}_crop_start"] = crop_start
                data[f"{key}_crop_end"] = crop_end
        return data


class RandSpatialCropd(RandSpatialCropd):
    def __call__(self, data, lazy: bool | None = None):
        data = super().__call__(data, lazy=lazy)
        # no need to iterate, all keys will be cropped the same way
        key = self.first_key(data)
        slices = self.cropper._slices
        crop_start = Tensor([s.start for s in slices])
        crop_end = Tensor([s.stop for s in slices])
        for key in self.key_iterator(data):
            if (
                f"{key}_crop_start" in data
            ):  # if we stack random cropping after a center or foreground crop, we need to adjust the crop_start and crop_end
                prev_crop_start = data[f"{key}_crop_start"]
                data[f"{key}_crop_end"] = prev_crop_start + crop_end
                data[f"{key}_crop_start"] = prev_crop_start + crop_start
            else:
                data[f"{key}_crop_start"] = crop_start
                data[f"{key}_crop_end"] = crop_end
        return data


class RandCropByPosNegLabeld(RandCropByPosNegLabeld):
    def __call__(self, data, lazy=None):
        d = dict(data)
        fg_indices = d.pop(self.fg_indices_key, None)
        bg_indices = d.pop(self.bg_indices_key, None)

        self.randomize(
            d.get(self.label_key), fg_indices, bg_indices, d.get(self.image_key)
        )

        # initialize returned list with shallow copy to preserve key ordering
        ret: list = [dict(d) for _ in range(self.cropper.num_samples)]
        # deep copy all the unmodified data
        for i in range(self.cropper.num_samples):
            for key in set(d.keys()).difference(set(self.keys)):
                ret[i][key] = deepcopy(d[key])

        lazy_ = self.lazy if lazy is None else lazy
        for key in self.key_iterator(d):
            for i, im in enumerate(self.cropper(d[key], randomize=False, lazy=lazy_)):
                ret[i][key] = im

                if (
                    f"{key}_crop_start" in d
                ):  # if we stack random cropping after a center or foreground crop, we need to adjust the crop_start and crop_end
                    prev_crop_start = d[f"{key}_crop_start"]
                    new_crop_start = (
                        prev_crop_start + Tensor(im.meta["crop_center"]) // 2
                    )
                    new_crop_end = new_crop_start + Tensor(self.cropper.spatial_size)
                else:
                    new_crop_start = Tensor([0, 0, 0])
                    new_crop_end = Tensor(self.cropper.spatial_size)
                ret[i][f"{key}_crop_start"] = new_crop_start
                ret[i][f"{key}_crop_end"] = new_crop_end

        if isinstance(ret, list):
            ret = ret[0]
        
        return ret


class ZeroOutChanneld(MapTransform):
    def __init__(
        self,
        keys,
        allow_missing_keys: bool = False,
        channel_idx: Union[List, int] = 0,
        value: Union[int, float] = 0,
    ):
        super().__init__(keys)
        self.allow_missing_keys = allow_missing_keys
        self.channel_idx = (
            [channel_idx] if isinstance(channel_idx, int) else channel_idx
        )
        self.value = value

    def __call__(self, data, channel_idx=None):
        channel_idx = self.channel_idx if channel_idx is None else channel_idx
        d = dict(data)
        for key in self.keys:
            if key not in d:
                if allow_missing_keys_mode:
                    continue
                else:
                    raise KeyError(f"Specified key '{key}' not found in data.")
            img = d[key]
            if img.shape[0] <= max(channel_idx):
                raise ValueError(
                    f"Channel index {channel_idx} out of bounds for image with shape {img.shape}"
                )
            for channel_idx in channel_idx:
                img[channel_idx] = self.value
            d[key] = img
        return d


class RandZeroOutChanneld(ZeroOutChanneld):
    def __init__(
        self, keys, channels: list, allow_missing_keys=False, value=0, prob: float = 0.5
    ):
        self.channels = channels
        self.prob = prob
        super().__init__(
            keys=keys,
            allow_missing_keys=allow_missing_keys,
            channel_idx=None,
            value=value,
        )

    def __call__(self, data):
        if random.random() > self.prob:
            return data

        channel_idx = random.randint(0, len(self.channels) - 1)
        loo_task = self.channels[channel_idx]
        d = super()(data, channel_idx)
        d["loo_task"] = loo_task
        return d
