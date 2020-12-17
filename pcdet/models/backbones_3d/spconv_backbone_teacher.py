from functools import partial

import spconv
import torch.nn as nn
import torch

from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ...utils import common_utils

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m

def get_same_indices(high_res_indices, low_res_indices):
    """
        Input should be long type
    """
    combined_indices_unique = torch.cat((high_res_indices,low_res_indices),0).unique(sorted=True,return_inverse=True,return_counts=True, dim=0) 
    combined_indices_inverse = combined_indices_unique[1][:high_res_indices.shape[0]]
    indices_sorted, sorted_index = torch.sort(combined_indices_inverse)
    combined_indices = torch.zeros(combined_indices_unique[2].shape[0], dtype=torch.long).cuda()
    combined_indices[indices_sorted] = sorted_index
    combined_indices_mask = combined_indices_unique[2]==2
    same_indices_high = combined_indices[torch.arange(0,combined_indices_unique[2].shape[0])[combined_indices_mask]]
    diff_indices = combined_indices[torch.arange(0,combined_indices_unique[2].shape[0])[~combined_indices_mask]]
    
    low_inverse = combined_indices_unique[1][high_res_indices.shape[0]:]
    low_sorted, low_sorted_indices = torch.sort(low_inverse)
    low_indices = torch.zeros(combined_indices_unique[2].shape[0], dtype=torch.long).cuda()
    low_indices[low_sorted] = low_sorted_indices
    same_indices_low = low_indices[torch.arange(0,combined_indices_unique[2].shape[0])[combined_indices_mask]]
    
    return same_indices_high, same_indices_low, diff_indices

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity.features
        out.features = self.relu(out.features)

        return out

class SparseBasicBlock_sub(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock_sub, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride
        self.sub_indices = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        if self.sub_indices is not None:
            same_indices_high, same_indices_low, diff_indices = get_same_indices(out.indices.long(), self.sub_indices.long())
            out.features[diff_indices] = 0

        out = self.conv2(out)
        out.features = self.bn2(out.features)
        if self.downsample is not None:
            identity = self.downsample(x)


        out.features += identity.features
        out.features = self.relu(out.features)

        return out

class SparseBasicBlock_SA(spconv.SparseModule):
    expansion = 1

    def __init__(
        self, 
        inplanes, 
        planes, 
        stride=1, 
        norm_fn=None, 
        downsample=None, 
        indice_key=None, 
        voxel_size=None, 
        point_cloud_range=None,
        SA_cfg_src=None,
        reset_extra_feature=False):
        super(SparseBasicBlock_SA, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        # self.conv2 = spconv.SubMConv3d(
        #     planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        # )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride
        self.sub_indices = None
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.downsample_times_map = SA_cfg_src.DOWNSAMPLE_FACTOR
        mlps = SA_cfg_src.MLPS
        for k in range(len(mlps)):
            mlps[k] = [mlps[k][0]] + mlps[k]

        self.SA_layer = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg_src.POOL_RADIUS,
                nsamples=SA_cfg_src.NSAMPLE,
                mlps=mlps,
                use_xyz=False,
                pool_method='max_pool',
            )
        self.reset_extra_feature = reset_extra_feature
        # self.conv_sa = nn.Conv2d(
        #     planes*len(SA_cfg_src.POOL_RADIUS), planes, kernel_size=1)
        # )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        if self.sub_indices is not None:
            # print("sub_indices:",self.sub_indices.shape)

            # print("sub_indices:",self.sub_indices)
            # print("out.batch_size:",out.batch_size)

            new_xyz = common_utils.get_voxel_centers(
                self.sub_indices[:, 1:4],
                downsample_times=self.downsample_times_map,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )

            new_xyz_batch_cnt = torch.LongTensor([(self.sub_indices[:,0]==i).float().sum() for i in range(out.batch_size)]).int().cuda()


            cur_coords = out.indices
            xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=self.downsample_times_map,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            # xyz_batch_cnt = torch.LongTensor([(cur_coords[:,0]==i).float().sum() for i in range(out.batch_size)]).int().cuda()
            xyz_batch_cnt = xyz.new_zeros(out.batch_size).int()
            for bs_idx in range(out.batch_size):
                xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

            pooled_points, pooled_features = self.SA_layer(
                xyz=xyz,
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=out.features.contiguous(),
            )
            same_indices_high, same_indices_low, diff_indices = get_same_indices(out.indices.long(), self.sub_indices.long())

            if self.reset_extra_feature:
                out.features[diff_indices] = 0
                out.features[same_indices_high] = pooled_features[same_indices_low]
                out.features = self.bn2(out.features)
            else:
                out.features[same_indices_high] = out.features.clone()[same_indices_high] + pooled_features[same_indices_low]
                out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)


        out.features += identity.features
        out.features = self.relu(out.features)

        return out

class VoxelResBackBone8x_SA(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        SA_cfg = self.model_cfg.SA_LAYER
        reset_extra_feature = self.model_cfg.get('RESET_EXTRA_FEATURE', False)

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock_SA(16, 16, norm_fn=norm_fn, indice_key='res1',
                SA_cfg_src=SA_cfg['x_conv1'], voxel_size=kwargs['voxel_size'], 
                point_cloud_range=kwargs['point_cloud_range'], reset_extra_feature=reset_extra_feature),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            # SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock_SA(32, 32, norm_fn=norm_fn, indice_key='res2',
                SA_cfg_src=SA_cfg['x_conv2'], voxel_size=kwargs['voxel_size'], 
                point_cloud_range=kwargs['point_cloud_range'], reset_extra_feature=reset_extra_feature),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            # SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock_SA(64, 64, norm_fn=norm_fn, indice_key='res3',
                SA_cfg_src=SA_cfg['x_conv3'], voxel_size=kwargs['voxel_size'], 
                point_cloud_range=kwargs['point_cloud_range'], reset_extra_feature=reset_extra_feature),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            # SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock_SA(128, 128, norm_fn=norm_fn, indice_key='res4',
                SA_cfg_src=SA_cfg['x_conv4'], voxel_size=kwargs['voxel_size'], 
                point_cloud_range=kwargs['point_cloud_range'], reset_extra_feature=reset_extra_feature),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        if 'sub_multi_scale_3d_features' in batch_dict:
            sub_filter = True
        else:
            sub_filter = False

        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)
        
        for k, module in self.conv1._modules.items():
            module.sub_indices = batch_dict['sub_multi_scale_3d_features']['x_conv1'].indices if sub_filter else None
        
        for k in ['1', '2']:
            self.conv2._modules[k].sub_indices = batch_dict['sub_multi_scale_3d_features']['x_conv2'].indices if sub_filter else None
            self.conv3._modules[k].sub_indices = batch_dict['sub_multi_scale_3d_features']['x_conv3'].indices if sub_filter else None
            self.conv4._modules[k].sub_indices = batch_dict['sub_multi_scale_3d_features']['x_conv4'].indices if sub_filter else None

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)


        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })


        return batch_dict

class VoxelResBackBone8x_SP(nn.Module):
    """
        sample pool
    """
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        SA_cfg = self.model_cfg.SA_LAYER

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock_SA(16, 16, norm_fn=norm_fn, indice_key='res1',
                SA_cfg_src=SA_cfg['x_conv1'], voxel_size=kwargs['voxel_size'], point_cloud_range=kwargs['point_cloud_range']),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            # SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock_SA(32, 32, norm_fn=norm_fn, indice_key='res2',
                SA_cfg_src=SA_cfg['x_conv2'], voxel_size=kwargs['voxel_size'], point_cloud_range=kwargs['point_cloud_range']),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            # SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock_SA(64, 64, norm_fn=norm_fn, indice_key='res3',
                SA_cfg_src=SA_cfg['x_conv3'], voxel_size=kwargs['voxel_size'], point_cloud_range=kwargs['point_cloud_range']),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            # SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock_SA(128, 128, norm_fn=norm_fn, indice_key='res4',
                SA_cfg_src=SA_cfg['x_conv4'], voxel_size=kwargs['voxel_size'], point_cloud_range=kwargs['point_cloud_range']),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        if 'sub_multi_scale_3d_features' in batch_dict:
            sub_filter = True
        else:
            sub_filter = False

        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)
        
        for k, module in self.conv1._modules.items():
            module.sub_indices = batch_dict['sub_multi_scale_3d_features']['x_conv1'].indices if sub_filter else None
        
        for k in ['1', '2']:
            self.conv2._modules[k].sub_indices = batch_dict['sub_multi_scale_3d_features']['x_conv2'].indices if sub_filter else None
            self.conv3._modules[k].sub_indices = batch_dict['sub_multi_scale_3d_features']['x_conv3'].indices if sub_filter else None
            self.conv4._modules[k].sub_indices = batch_dict['sub_multi_scale_3d_features']['x_conv4'].indices if sub_filter else None

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)


        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })


        return batch_dict
