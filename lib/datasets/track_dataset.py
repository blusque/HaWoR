import copy
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, ToTensor, Compose
import numpy as np
import cv2

from hawor.utils.process import run_mano, run_mano_left
from hawor.utils.render_openpose import render_openpose
from lib.core import constants
from lib.eval_utils.custom_utils import cam_to_img
from lib.utils.imutils import crop, boxes_2_cs, crop_j2d
import os
from PIL import Image

class TrackDataset(Dataset):
    """
    Track Dataset Class - Load images/crops of the tracked boxes.
    """
    def __init__(self, imgfiles, boxes, crop_size=256, dilate=1.0,
                img_focal=None, img_center=None, normalization=True, for_preprocess=False,
                item_idx=0):
        super(TrackDataset, self).__init__()

        self.imgfiles = imgfiles
        self.crop_size = crop_size
        self.normalization = normalization
        self.normalize_img = Compose([
                            ToTensor(),
                            Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
                        ])

        self.boxes = boxes
        self.box_dilate = dilate
        self.centers, self.scales = boxes_2_cs(boxes)

        self.img_focal = img_focal
        self.img_center = img_center
        self.for_preprocess = for_preprocess
        self.item_idx = item_idx


    def __len__(self):
        return len(self.imgfiles)
    
    
    def __getitem__(self, index):
        item = {}
        imgfile = self.imgfiles[index]
        scale = self.scales[index] * self.box_dilate
        center = self.centers[index]
        img_focal = self.img_focal
        img_center = self.img_center

        img = cv2.imread(imgfile)[:,:,::-1]
        img_crop = crop(img, center, scale, 
                        [self.crop_size, self.crop_size], 
                        rot=0).astype('uint8')

        if self.for_preprocess:
            model_root = os.path.dirname(os.path.dirname(imgfile))
            img_crop_name = os.path.join(model_root, 'img_crop', str(self.item_idx), f"{index}.png")
            if not os.path.exists(os.path.dirname(img_crop_name)):
                os.makedirs(os.path.dirname(img_crop_name))
            Image.fromarray(img_crop).save(img_crop_name)
            item['img_crop_path'] = img_crop_name
    
        else:
            if self.normalization:
                img_crop = self.normalize_img(img_crop)
            else:
                img_crop = torch.from_numpy(img_crop)
            item['img'] = img_crop

        if self.img_focal is None:
            orig_shape = img.shape[:2]
            img_focal = self.est_focal(orig_shape)

        if self.img_center is None:
            orig_shape = img.shape[:2]
            img_center = self.est_center(orig_shape)

        
        item['img_idx'] = torch.tensor(index).long()
        item['scale'] = torch.tensor(scale).float()
        item['center'] = torch.tensor(center).float()
        item['img_focal'] = torch.tensor(img_focal).float()
        item['img_center'] = torch.tensor(img_center).float()

        return item


    def est_focal(self, orig_shape):
        h, w = orig_shape
        focal = np.sqrt(h**2 + w**2)
        return focal

    def est_center(self, orig_shape):
        h, w = orig_shape
        center = np.array([w/2., h/2.])
        return center

class TrackDatasetTrain(Dataset):
    """
    Track Dataset Class - Load images/crops of the tracked boxes.
    """
    def __init__(self, imgfiles, boxes, 
                 _cam_betas, _cam_full_pose, _cam_j3d, _cam_j2d, _j3d_wo_trans,
                 crop_size=256, dilate=1.0,
                img_focal=None, img_center=None, normalization=True, for_preprocess=False,
                item_idx=0, vis_img_crop_j2d=False, do_flip=False):
        super(TrackDatasetTrain, self).__init__()

        self.imgfiles = imgfiles
        self.crop_size = crop_size
        self.normalization = normalization
        self.normalize_img = Compose([
                            ToTensor(),
                            Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
                        ])

        self.boxes = boxes
        self.box_dilate = dilate
        self.centers, self.scales = boxes_2_cs(boxes)

        self.img_focal = img_focal
        self.img_center = img_center
        self.for_preprocess = for_preprocess
        self.item_idx = item_idx

        self._cam_betas = _cam_betas
        self._cam_full_pose = _cam_full_pose
        self._cam_j3d = _cam_j3d
        self._j3d_wo_trans = _j3d_wo_trans
        self._cam_j2d = _cam_j2d
        self.vis_img_crop_j2d = vis_img_crop_j2d
        self.do_flip = do_flip


    def __len__(self):
        return len(self.imgfiles)
    
    
    def __getitem__(self, index):
        item = {}
        imgfile = self.imgfiles[index]
        scale = self.scales[index] * self.box_dilate
        center = self.centers[index]

        _cam_betas = self._cam_betas[index]
        _cam_full_pose = self._cam_full_pose[index]
        _cam_j3d = self._cam_j3d[index]
        _j3d_wo_trans = self._j3d_wo_trans[index]
        _cam_j2d = self._cam_j2d[index]

        img_focal = self.img_focal
        img_center = self.img_center

        img = cv2.imread(imgfile)[:,:,::-1]
        if self.do_flip:
            img = img[:, ::-1, :]
            img_width = img.shape[1]
            center[0] = img_width - center[0] - 1
            _cam_j2d[:, 0] = img_width - _cam_j2d[:, 0] - 1
        img_crop = crop(img, center, scale, 
                        [self.crop_size, self.crop_size], 
                        rot=0).astype('uint8')

        # TODO: crop cam_j2d and normalize
        _cam_j2d_crop = crop_j2d(_cam_j2d, center, scale, 
                        [self.crop_size, self.crop_size], 
                        rot=0)
        if self.do_flip:
            _j3d_wo_trans[:, 0] = - _j3d_wo_trans[:, 0]
            _cam_full_pose[1::3] *= -1
            _cam_full_pose[2::3] *= -1

            # zero_trans = _cam_full_pose[:3] * 0
            # outputs = run_mano(zero_trans[None, None], _cam_full_pose[None, None, :3], _cam_full_pose[None, None, 3:], None, _cam_betas[None, None])
            # j3d_debug = outputs["joints"].cpu()[0, 0]
            # # _j3d_wo_trans should be equal to j3d_debug, check the error
            # print(abs(_j3d_wo_trans - j3d_debug).mean())
        _cam_j2d_crop = _cam_j2d_crop / self.crop_size - 0.5 # [-0.5, 0.5]

        if self.vis_img_crop_j2d:
            img_crop_j2d = copy.deepcopy(img_crop)
            gt_keypoints = 256 * (_cam_j2d_crop.numpy() + 0.5)
            gt_keypoints = np.concatenate((gt_keypoints, np.ones_like(gt_keypoints)[:, [0]]), axis=-1)
            gt_keypoints_img = render_openpose(img_crop_j2d, gt_keypoints)
            model_root = os.path.dirname(os.path.dirname(imgfile))
            img_crop_name = os.path.join(model_root, 'img_crop_j2d', str(self.item_idx), f"{index}.png")
            if not os.path.exists(os.path.dirname(img_crop_name)):
                os.makedirs(os.path.dirname(img_crop_name))
            Image.fromarray(gt_keypoints_img).save(img_crop_name)

        if self.for_preprocess:
            video_root = os.path.dirname(os.path.dirname(imgfile))
            video_name = os.path.basename(video_root)
            dataset_root = os.path.dirname(video_root)
            img_crop_name = os.path.join(video_name, 'img_crop_sequences', str(self.item_idx), f"{index}.png")
            img_crop_full_path = os.path.join(dataset_root, img_crop_name)
            if not os.path.exists(os.path.dirname(img_crop_full_path)):
                os.makedirs(os.path.dirname(img_crop_full_path))
            Image.fromarray(img_crop).save(img_crop_full_path)
            item['img_crop_path'] = img_crop_name
    
        else:
            if self.normalization:
                img_crop = self.normalize_img(img_crop)
            else:
                img_crop = torch.from_numpy(img_crop)
            item['img'] = img_crop
        
        item['img_idx'] = torch.tensor(index).long()
        item['scale'] = torch.tensor(scale).float()
        item['center'] = torch.tensor(center).float()
        item['img_focal'] = torch.tensor(img_focal).float()
        item['img_center'] = torch.tensor(img_center).float()

        # GT
        item['gt_cam_betas'] = _cam_betas
        item['gt_cam_full_pose'] = _cam_full_pose
        item['gt_j3d_wo_trans'] = _j3d_wo_trans
        item['gt_cam_j2d'] = _cam_j2d_crop

        return item


    def est_focal(self, orig_shape):
        h, w = orig_shape
        focal = np.sqrt(h**2 + w**2)
        return focal

    def est_center(self, orig_shape):
        h, w = orig_shape
        center = np.array([w/2., h/2.])
        return center


class TrackDatasetEval(Dataset):
    """
    Track Dataset Class - Load images/crops of the tracked boxes.
    """
    def __init__(self, imgfiles, boxes, 
                 crop_size=256, dilate=1.0,
                img_focal=None, img_center=None, normalization=True,
                item_idx=0, do_flip=False):
        super(TrackDatasetEval, self).__init__()

        self.imgfiles = imgfiles
        self.crop_size = crop_size
        self.normalization = normalization
        self.normalize_img = Compose([
                            ToTensor(),
                            Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
                        ])

        self.boxes = boxes
        self.box_dilate = dilate
        self.centers, self.scales = boxes_2_cs(boxes)

        self.img_focal = img_focal
        self.img_center = img_center
        self.item_idx = item_idx
        self.do_flip = do_flip

    def __len__(self):
        return len(self.imgfiles)
    
    
    def __getitem__(self, index):
        item = {}
        imgfile = self.imgfiles[index]
        scale = self.scales[index] * self.box_dilate
        center = self.centers[index]

        img_focal = self.img_focal
        img_center = self.img_center

        img = cv2.imread(imgfile)[:,:,::-1]
        if self.do_flip:
            img = img[:, ::-1, :]
            img_width = img.shape[1]
            center[0] = img_width - center[0] - 1
        img_crop = crop(img, center, scale, 
                        [self.crop_size, self.crop_size], 
                        rot=0).astype('uint8')
        # cv2.imwrite('debug_crop.png', img_crop[:,:,::-1])
        
        if self.normalization:
            img_crop = self.normalize_img(img_crop)
        else:
            img_crop = torch.from_numpy(img_crop)
        item['img'] = img_crop

        if self.img_focal is None:
            orig_shape = img.shape[:2]
            img_focal = self.est_focal(orig_shape)

        if self.img_center is None:
            orig_shape = img.shape[:2]
            img_center = self.est_center(orig_shape)

        
        if self.do_flip:
            # center[0] = img_width - center[0] - 1 
            item['do_flip'] = torch.tensor(1).float()
        item['img_idx'] = torch.tensor(index).long()
        item['scale'] = torch.tensor(scale).float()
        item['center'] = torch.tensor(center).float()
        item['img_focal'] = torch.tensor(img_focal).float()
        item['img_center'] = torch.tensor(img_center).float()
        

        return item


    def est_focal(self, orig_shape):
        h, w = orig_shape
        focal = np.sqrt(h**2 + w**2)
        return focal

    def est_center(self, orig_shape):
        h, w = orig_shape
        center = np.array([w/2., h/2.])
        return center

