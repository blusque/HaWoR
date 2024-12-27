import argparse
import copy
import os
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from cmib.data.lafan1_dataset import LAFAN1Dataset
from cmib.data.utils import flip_bvh, increment_path, process_seq_names
from cmib.model.network import TransformerModel
from cmib.model.preprocess import (lerp_input_repr, replace_constant,
                                   slerp_input_repr, vectorize_representation)
from cmib.model.skeleton import (Skeleton, sk_joints_to_remove, sk_offsets, sk_parents, amass_offsets)
from torch.utils.data import Dataset
from hand_utils.geometry import rotmat_to_rot6d
from hand_utils.process import run_mano, run_mano_left
from hand_utils.rotation import angle_axis_to_quaternion, angle_axis_to_rotation_matrix, quaternion_to_rotation_matrix, rotation_matrix_to_angle_axis
from scipy.spatial.transform import Slerp, Rotation
import sys
from torch.utils.tensorboard import SummaryWriter
            

def slerp_interpolation_aa(pos, valid):

    B, T, N, _ = pos.shape  # B: 批次大小, T: 时间步长, N: 关节数, 4: 四元数维度
    pos_interp = pos.copy()  # 创建副本以存储插值结果
    
    for b in range(B):
        for n in range(N):
            quat_b_n = pos[b, :, n, :]
            valid_b_n = valid[b, :]
            
            invalid_idxs = np.where(~valid_b_n)[0]
            valid_idxs = np.where(valid_b_n)[0]

            if len(invalid_idxs) == 0:
                continue
            
            if len(valid_idxs) > 1:
                valid_times = valid_idxs  # 有效时间步
                valid_rots = Rotation.from_rotvec(quat_b_n[valid_idxs])  # 有效四元数
                
                slerp = Slerp(valid_times, valid_rots)
                
                for idx in invalid_idxs:
                    if idx < valid_idxs[0]:  # 时间步小于第一个有效时间步，进行外推
                        pos_interp[b, idx, n, :] = quat_b_n[valid_idxs[0]]  # 复制第一个有效四元数
                    elif idx > valid_idxs[-1]:  # 时间步大于最后一个有效时间步，进行外推
                        pos_interp[b, idx, n, :] = quat_b_n[valid_idxs[-1]]  # 复制最后一个有效四元数
                    else:
                        interp_rot = slerp([idx])
                        pos_interp[b, idx, n, :] = interp_rot.as_rotvec()[0]
    
    return pos_interp

def linear_interpolation_nd(pos, valid):
    B, T = pos.shape[:2]  # 取出批次大小B和时间步长T
    feature_dim = pos.shape[2]  # ** 代表的任意维度
    pos_interp = pos.copy()  # 创建一个副本，用来保存插值结果
    
    for b in range(B):
        for idx in range(feature_dim):  # 针对任意维度
            pos_b_idx = pos[b, :, idx]  # 取出第b批次对应的**维度下的一个时间序列
            valid_b = valid[b, :]  # 当前批次的有效标志
            
            # 找到无效的索引（False）
            invalid_idxs = np.where(~valid_b)[0]
            valid_idxs = np.where(valid_b)[0]

            if len(invalid_idxs) == 0:
                continue
            
            # 对无效部分进行线性插值
            if len(valid_idxs) > 1:  # 确保有足够的有效点用于插值
                pos_b_idx[invalid_idxs] = np.interp(invalid_idxs, valid_idxs, pos_b_idx[valid_idxs])
                pos_interp[b, :, idx] = pos_b_idx  # 保存插值结果
    
    return pos_interp

def world2canonical_convert(R_c2w_sla, t_c2w_sla, data_out, handedness):
    init_rot_mat = copy.deepcopy(data_out["init_root_orient"])
    init_rot_mat = torch.einsum("tij,btjk->btik", R_c2w_sla, init_rot_mat)
    init_rot = rotation_matrix_to_angle_axis(init_rot_mat)
    init_rot_quat = angle_axis_to_quaternion(init_rot)
    # data_out["init_root_orient"] = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
    # data_out["init_hand_pose"] = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
    data_out_init_root_orient = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
    data_out_init_hand_pose = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])

    init_trans = data_out["init_trans"] # (B, T, 3)
    if handedness == "left":
        outputs = run_mano_left(data_out["init_trans"], data_out_init_root_orient, data_out_init_hand_pose, betas=data_out["init_betas"])

    elif handedness == "right":
        outputs = run_mano(data_out["init_trans"], data_out_init_root_orient, data_out_init_hand_pose, betas=data_out["init_betas"])
    root_loc = outputs["joints"][..., 0, :].cpu()  # (B, T, 3)
    offset = init_trans - root_loc  # It is a constant, no matter what the rotation is.
    init_trans = (
        torch.einsum("tij,btj->bti", R_c2w_sla, root_loc)
        + t_c2w_sla[None, :]
        + offset
    )

    data_world = {
        "init_root_orient": init_rot, # (B, T, 3)
        "init_hand_pose": data_out_init_hand_pose, # (B, T, 15, 3)
        "init_trans": init_trans,  # (B, T, 3)
        "init_betas": data_out["init_betas"]  # (B, T, 10)
    }

    return data_world

def generate_mask(M, T, context_len):
    # 初始化 mask 矩阵，尺寸为 (M, T)
    mask = np.ones((M, T), dtype=bool)
    
    # 遍历每组数据
    for i in range(M):
        # 随机生成 mask 起始帧和长度
        start_frame = np.random.randint(context_len, T - context_len)  # 从第1帧到第T-2帧的随机位置
        max_length = T - context_len - start_frame
        mask_length = np.random.randint(1, max_length+1)  # [1,max_length]
        
        # 应用 mask
        print("mask", start_frame, "to", start_frame + mask_length - 1)
        mask[i, start_frame:start_frame + mask_length] = False
    
    return mask

def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

class Hot3dTrainDataset(Dataset):

    def __init__(self, root, seq_len, preprocess=False, preload=False, fix_start_context=False):
        super(Hot3dTrainDataset, self).__init__()
        self.root = root
        self.seq_paths = self.find_all_pth_files()
        self.num_joints = 15
        self.seq_len = seq_len
        if preprocess:
            for i in tqdm(range(len(self.seq_paths))):
                self.preprocess(i)
            sys.exit()
        self.preload = preload
        self.fix_start_context = fix_start_context
        if self.preload:
            self.data = []
            print("preload data ...")
            for i in tqdm(range(len(self.seq_paths))):
                self.data.append(joblib.load(self.seq_paths[i]))
    
    def find_all_pth_files(self):
        pth_files = []
        # 使用 os.walk 递归遍历目录
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith('.pth'):
                    # 拼接完整路径
                    full_path = os.path.join(root, file)
                    pth_files.append(full_path)
        return pth_files

    def __len__(self):
        return len(self.seq_paths)
    
    def preprocess(self, index):
        block_print()
        data = joblib.load(self.seq_paths[index])
        item = copy.deepcopy(data)

        num_joints = 15

        global_trans = item['trans'] # (2, seq_len, 3)
        global_rot = item['rot'] #(2, seq_len, 3)
        hand_pose = item['hand_pose'] # (2, seq_len, 45)
        betas = item['betas'] # (2, seq_len, 10)
        valid = item['valid'] # (2, seq_len)

        N, T, _ = global_trans.shape

        if valid.sum() == N*T:
            generate_mask(N, T, context_len=10)
            print("random masking")
        R_canonical2world_left_aa = torch.from_numpy(global_rot[0, 0])
        R_canonical2world_right_aa = torch.from_numpy(global_rot[1, 0])
        R_world2canonical_left = angle_axis_to_rotation_matrix(R_canonical2world_left_aa).t()
        R_world2canonical_right = angle_axis_to_rotation_matrix(R_canonical2world_right_aa).t()
        

        # transform left hand to canonical
        hand_pose = hand_pose.reshape(N, T, num_joints, 3)
        data_world_left = {
            "init_trans": torch.from_numpy(global_trans[0:1]),
            "init_root_orient": angle_axis_to_rotation_matrix(torch.from_numpy(global_rot[0:1])),
            "init_hand_pose": angle_axis_to_rotation_matrix(torch.from_numpy(hand_pose[0:1])),
            "init_betas": torch.from_numpy(betas[0:1]),
        }

        data_left_init_root_orient = rotation_matrix_to_angle_axis(data_world_left["init_root_orient"])
        data_left_init_hand_pose = rotation_matrix_to_angle_axis(data_world_left["init_hand_pose"])
        outputs = run_mano_left(data_world_left["init_trans"], data_left_init_root_orient, data_left_init_hand_pose, betas=data_world_left["init_betas"])
        init_trans = data_world_left["init_trans"][0, 0] # (3,)
        root_loc = outputs["joints"][0, 0, 0, :].cpu()  # (3,)
        offset = init_trans - root_loc  # It is a constant, no matter what the rotation is.
        t_world2canonical_left = -torch.einsum("ij,j->i", R_world2canonical_left, root_loc) - offset

        R_world2canonical_left = R_world2canonical_left.repeat(T, 1, 1)
        t_world2canonical_left = t_world2canonical_left.repeat(T, 1)
        data_canonical_left = world2canonical_convert(R_world2canonical_left, t_world2canonical_left, data_world_left, "left")

        # transform right hand to canonical
        data_world_right = {
            "init_trans": torch.from_numpy(global_trans[1:2]),
            "init_root_orient": angle_axis_to_rotation_matrix(torch.from_numpy(global_rot[1:2])),
            "init_hand_pose": angle_axis_to_rotation_matrix(torch.from_numpy(hand_pose[1:2])),
            "init_betas": torch.from_numpy(betas[1:2]),
        }

        data_right_init_root_orient = rotation_matrix_to_angle_axis(data_world_right["init_root_orient"])
        data_right_init_hand_pose = rotation_matrix_to_angle_axis(data_world_right["init_hand_pose"])
        outputs = run_mano(data_world_right["init_trans"], data_right_init_root_orient, data_right_init_hand_pose, betas=data_world_right["init_betas"])
        init_trans = data_world_right["init_trans"][0, 0] # (3,)
        root_loc = outputs["joints"][0, 0, 0, :].cpu()  # (3,)
        offset = init_trans - root_loc  # It is a constant, no matter what the rotation is.
        t_world2canonical_right = -torch.einsum("ij,j->i", R_world2canonical_right, root_loc) - offset

        R_world2canonical_right = R_world2canonical_right.repeat(T, 1, 1)
        t_world2canonical_right = t_world2canonical_right.repeat(T, 1)
        data_canonical_right = world2canonical_convert(R_world2canonical_right, t_world2canonical_right, data_world_right, "right")

        # merge left and right canonical data
        global_rot = torch.cat((data_canonical_left['init_root_orient'], data_canonical_right['init_root_orient']))
        global_trans = torch.cat((data_canonical_left['init_trans'], data_canonical_right['init_trans'])).numpy()
        
        # global_rot = angle_axis_to_quaternion(global_rot).numpy().reshape(N, T, 1, 4)
        global_rot = global_rot.reshape(N, T, 1, 3).numpy()
        
        hand_pose = hand_pose.reshape(N, T, 15, 3)
        # hand_pose = angle_axis_to_quaternion(torch.from_numpy(hand_pose)).numpy()

        # lerp and slerp
        global_trans_lerped = linear_interpolation_nd(global_trans, valid)
        betas_lerped = linear_interpolation_nd(betas, valid)
        global_rot_slerped = slerp_interpolation_aa(global_rot, valid)
        hand_pose_slerped = slerp_interpolation_aa(hand_pose, valid)
        

        # convert to rot6d
        global_rot_mat = angle_axis_to_rotation_matrix(torch.from_numpy(global_rot.reshape(N*T, -1)))
        global_rot_rot6d = rotmat_to_rot6d(global_rot_mat).reshape(N, T, -1).numpy()
        hand_pose_mat = angle_axis_to_rotation_matrix(torch.from_numpy(hand_pose.reshape(N*T*self.num_joints, -1)))
        hand_pose_rot6d = rotmat_to_rot6d(hand_pose_mat).reshape(N, T, -1).numpy()

        global_rot_slerped_mat = angle_axis_to_rotation_matrix(torch.from_numpy(global_rot_slerped.reshape(N*T, -1)))
        # global_rot_slerped_mat = quaternion_to_rotation_matrix(torch.from_numpy(global_rot_slerped.reshape(N*T, -1)))
        global_rot_slerped_rot6d = rotmat_to_rot6d(global_rot_slerped_mat).reshape(N, T, -1).numpy()
        hand_pose_slerped_mat = angle_axis_to_rotation_matrix(torch.from_numpy(hand_pose_slerped.reshape(N*T*num_joints, -1)))
        # hand_pose_slerped_mat = quaternion_to_rotation_matrix(torch.from_numpy(hand_pose_slerped.reshape(N*T*num_joints, -1)))
        hand_pose_slerped_rot6d = rotmat_to_rot6d(hand_pose_slerped_mat).reshape(N, T, -1).numpy()


        # concat to (T, concat_dim)
        global_pose_vec_gt = np.concatenate((global_trans, betas, global_rot_rot6d, hand_pose_rot6d), axis=-1).transpose(1, 0, 2).reshape(T, -1)
        global_pose_vec_input = np.concatenate((global_trans_lerped, betas_lerped, global_rot_slerped_rot6d, hand_pose_slerped_rot6d), axis=-1).transpose(1, 0, 2).reshape(T, -1)

        assert len(global_pose_vec_input) == self.seq_len
        assert len(global_pose_vec_gt) == self.seq_len
        enable_print()
        processed_data = {
            "input": global_pose_vec_input,
            "gt": global_pose_vec_gt,
            "seq_valid": valid
        }
        save_path = self.seq_paths[index].replace("datasets_filling_net", "datasets_filling_net_processed")
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        joblib.dump(processed_data, save_path)

    
    def __getitem__(self, index):
        if self.preload:
            data = self.data[index]
        else:
            data = joblib.load(self.seq_paths[index])
        # data = self.preprocess(index)
        input = data['input']
        gt = data['gt']
        seq_valid = data['seq_valid']

        if self.fix_start_context:
            align_input = np.zeros_like(input)
            align_gt = np.zeros_like(gt)
            align_valid = np.zeros_like(seq_valid)

            first_false_index = len(gt) - 1
            for i, row in enumerate(seq_valid[:, :]):
                false_indices = np.nonzero(~row)[0]
                if len(false_indices) > 0:
                    first_false_index = min(false_indices[0].item(), first_false_index)
            align_input[:(len(gt) - (first_false_index-10))] = input[first_false_index-10:]
            align_gt[:(len(gt) - (first_false_index-10))] = gt[first_false_index-10:]
            align_valid[:, :(len(gt) - (first_false_index-10))] = seq_valid[:, first_false_index-10:]

            input = align_input
            gt = align_gt
            seq_valid = align_valid


        if seq_valid[:, -10:].sum() != 20:
            # print(seq_valid)
            last_valid_index = len(gt) - 1
            for i, row in enumerate(seq_valid[:, :-10]):
                # 获取每行中为 True 的索引
                true_indices = np.nonzero(row)[0]
                # 检查是否存在 True，若存在则设置其后的元素为 True
                if len(true_indices) > 0:
                    last_true_idx = true_indices[-1].item()
                    seq_valid[i, last_true_idx:] = True  # 将倒数第一个 True 之后的所有元素设为 True
                    last_valid_index = min(last_valid_index, last_true_idx)
            gt[last_valid_index+1:] = gt[last_valid_index]
            input[last_valid_index+1:] = input[last_valid_index]
        
        return input, gt, seq_valid

def train(opt, device):

    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    if opt.epochs:
        epochs = opt.epochs
        max_iters = -1
    else:
        epochs = opt.iters // opt.batch_size + 1
        max_iters = opt.iters
    save_interval = opt.save_interval
                              
    horizon = opt.seq_len
    print(f"Horizon: {horizon}")
    print(f"Horizon with Conditioning: {horizon}")

    hand_dataset_dir = opt.hand_dataset_dir
    hand_dataset = Hot3dTrainDataset(hand_dataset_dir, opt.seq_len, opt.preprocess, opt.preload, opt.fix_start_context)

    # tensor_dataset = TensorDataset(global_pose_vec_input, global_pose_vec_gt, seq_labels) # (B,seq_len,22*3+22*4)
    lafan_data_loader = DataLoader(hand_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    pos_dim = 3
    shape_dim = 10
    rot_dim = (hand_dataset.num_joints + 1) * 6 # rot6d
    repr_dim = 2 * (pos_dim + shape_dim + rot_dim)
    nhead = 8 # repr_dim = 154

    transformer_encoder = TransformerModel(seq_len=horizon, input_dim=repr_dim, d_model=384, nhead=nhead, d_hid=2048, nlayers=8, dropout=0.05, out_dim=repr_dim, masked_attention_stage=opt.masked_attention_stage)
    transformer_encoder.to(device)

    # weight_path = "/media/zjl/Disk16T/Conditional-Motion-In-Betweening/runs/train/cmib_12011/weights/train-5000.pt"
    # ckpt = torch.load(weight_path, map_location=device)
    # transformer_encoder.load_state_dict(ckpt['transformer_encoder_state_dict'])
    # transformer_encoder.train()

    l1_loss = nn.L1Loss()
    optim = AdamW(params=transformer_encoder.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.9)

    if os.path.exists(opt.resume):
        ckpt = torch.load(opt.resume, map_location=device)
        transformer_encoder.load_state_dict(ckpt['transformer_encoder_state_dict'])
        resume_epoch = ckpt['epoch']
        optim.load_state_dict(ckpt['optimizer_state_dict'])
    else:
        resume_epoch = 0

    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb"))
    idx = 0
    iter_idx = 0

    for epoch in range(1, epochs + 1):

        if epoch <= resume_epoch:
            print(f'skip epoch {epoch}')
            continue

        pbar = tqdm(lafan_data_loader, position=1, desc="Batch")

        recon_shape_loss = []
        recon_pos_loss = []
        recon_rot_loss = []
        total_loss_list = []
        

        for pose_interpolated_input, minibatch_pose_gt, seq_valid in pbar:

            B, T, _ = pose_interpolated_input.shape
            pose_interpolated_input = pose_interpolated_input.permute(1,0,2)

            src_mask = torch.zeros((horizon, horizon), device=device).type(torch.bool)
            src_mask = src_mask.to(device)
            pose_interpolated_input = pose_interpolated_input.to(device)
            
            
            if opt.masked_attention_stage:
                valid = seq_valid.all(dim=1).permute(1, 0) # (T,B)
                valid_atten = seq_valid.all(dim=1).unsqueeze(1) # (B,1,T)
                data_mask = torch.zeros((horizon, B, 1), device=device, dtype=pose_interpolated_input.dtype)
                data_mask[valid] = 1
                atten_mask = torch.ones((B, 1, horizon),
                            device=device, dtype=torch.bool)
                atten_mask[valid_atten] = False
                atten_mask = atten_mask.unsqueeze(2).repeat(1, 1, T, 1) # (B,1,T,T)

                output = transformer_encoder(pose_interpolated_input, src_mask, data_mask, atten_mask)
            else:
                output = transformer_encoder(pose_interpolated_input, src_mask)

            output = output.permute(1,0,2).reshape(B, T, 2, -1) # 2 = two hands
            minibatch_pose_gt = minibatch_pose_gt.reshape(B, T, 2, -1).to(device)

            if opt.masked_loss:
                loss_part = ~seq_valid.all(dim=1) # (B,T)
                pos_pred = output[loss_part][...,:pos_dim]
                pos_gt = minibatch_pose_gt[loss_part][...,:pos_dim]
                pos_loss = l1_loss(pos_pred, pos_gt)
                recon_pos_loss.append(opt.loss_pos_weight * pos_loss)

                shape_pred = output[loss_part][...,pos_dim:pos_dim+shape_dim]
                shape_gt = minibatch_pose_gt[loss_part][...,pos_dim:pos_dim+shape_dim]
                shape_loss = l1_loss(shape_pred, shape_gt)
                recon_shape_loss.append(opt.loss_shape_weight * shape_loss)

                rot_pred = output[loss_part][...,pos_dim+shape_dim:]
                rot_gt = minibatch_pose_gt[loss_part][...,pos_dim+shape_dim:]
                rot_loss = l1_loss(rot_pred, rot_gt)
                recon_rot_loss.append(opt.loss_rot_weight * rot_loss)
            else:
                pos_pred = output[...,:pos_dim]
                pos_gt = minibatch_pose_gt[...,:pos_dim]
                pos_loss = l1_loss(pos_pred, pos_gt)
                recon_pos_loss.append(opt.loss_pos_weight * pos_loss)

                shape_pred = output[...,pos_dim:pos_dim+shape_dim]
                shape_gt = minibatch_pose_gt[...,pos_dim:pos_dim+shape_dim]
                shape_loss = l1_loss(shape_pred, shape_gt)
                recon_shape_loss.append(opt.loss_shape_weight * shape_loss)

                rot_pred = output[...,pos_dim+shape_dim:]
                rot_gt = minibatch_pose_gt[...,pos_dim+shape_dim:]
                rot_loss = l1_loss(rot_pred, rot_gt)
                recon_rot_loss.append(opt.loss_rot_weight * rot_loss)

            total_g_loss = opt.loss_pos_weight * pos_loss + \
                            opt.loss_rot_weight * rot_loss + \
                            opt.loss_shape_weight * shape_loss
            total_loss_list.append(total_g_loss)
            # print("")
            # print("loss:", opt.loss_pos_weight * pos_loss.item(), opt.loss_rot_weight * rot_loss.item(), opt.loss_shape_weight * shape_loss.item())
            # print(rot_pred[0,0,0,:6])
            # print(rot_gt[0,0,0,:6])
            tb_dict = {
                "Train/Loss/Shape Loss": opt.loss_shape_weight * shape_loss.item(), 
                "Train/Loss/Position Loss": opt.loss_pos_weight * pos_loss.item(), 
                "Train/Loss/Rotation Loss": opt.loss_rot_weight * rot_loss.item(),
                "Train/Loss/Total Loss": total_g_loss.item(),
            }
            for k, v in tb_dict.items():
                writer.add_scalar(k, v, global_step=idx)
            idx += 1
        
            optim.zero_grad()
            total_g_loss.backward()
            # torch.nn.utils.clip_grad_norm_(transformer_encoder.parameters(), 1.0, error_if_nonfinite=False)
            optim.step()
            if max_iters > 0 and iter_idx >= max_iters:
                break

        scheduler.step()

        # Log
        log_dict = {
            "Train/Loss/Shape Loss": torch.stack(recon_shape_loss).mean().item(), 
            "Train/Loss/Position Loss": torch.stack(recon_pos_loss).mean().item(), 
            "Train/Loss/Rotatation Loss": torch.stack(recon_rot_loss).mean().item(),
            "Train/Loss/Total Loss": torch.stack(total_loss_list).mean().item(),
        }
        print(f"epoch {epoch}")
        print(log_dict)

        # Save model
        if (epoch % save_interval) == 0 or (max_iters > 0 and iter_idx >= max_iters):
            ckpt = {'epoch': epoch,
                    'transformer_encoder_state_dict': transformer_encoder.state_dict(),
                    'horizon': transformer_encoder.seq_len,
                    'd_model': transformer_encoder.d_model,
                    'nhead': transformer_encoder.nhead,
                    'd_hid': transformer_encoder.d_hid,
                    'nlayers': transformer_encoder.nlayers,
                    'optimizer_state_dict': optim.state_dict(),
                    'loss': total_g_loss}
            torch.save(ckpt, os.path.join(wdir, f'train-{epoch}.pt'))
            print(f"[MODEL SAVED at {epoch} Epoch]")
        if max_iters > 0 and iter_idx >= max_iters:
            break

    torch.cuda.empty_cache()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--hand_dataset_dir', type=str, default='ubisoft-laforge-animation-dataset/output/BVH', help='BVH dataset path')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--iters', type=int, default=None)
    parser.add_argument('--device', default='0', help='cuda device')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--exp_name', default='exp', help='save to project/name')
    parser.add_argument('--save_interval', type=int, default=50, help='Log model after every "save_period" epoch')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='generator_learning_rate')
    parser.add_argument('--loss_shape_weight', type=float, default=0.05, help='loss_shape_weight')
    parser.add_argument('--loss_pos_weight', type=float, default=0.05, help='loss_pos_weight')
    parser.add_argument('--loss_rot_weight', type=float, default=2.0, help='loss_rot_weight')
    parser.add_argument('--seq_len', type=int, default=120, help='seq len')
    parser.add_argument('--preprocess', action="store_true")
    parser.add_argument('--preload', action="store_true")
    parser.add_argument('--masked_attention_stage', action="store_true")
    parser.add_argument('--masked_loss', action="store_true")
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--fix_start_context', action="store_true")
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    train(opt, device)
