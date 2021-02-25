from functools import partial

import spconv
import torch.nn as nn
import torch
from ...utils.mimic_utils import get_same_indices

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

def get_voxel_coords_inbox_dict(batch_dict):
    expand_index = torch.LongTensor([
                [0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 1, -1], [0, 1, 0, 1], [0, 1, 0, 0], [0, 1, 0, -1], [0, 1, -1, 1], [0, 1, -1, 0], [0, 1, -1, -1],
                [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 1, -1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 1], [0, 0, -1, 0], [0, 0, -1, -1],
                [0, -1, 1, 1], [0, -1, 1, 0], [0, -1, 1, -1], [0, -1, 0, 1], [0, -1, 0, 0], [0, -1, 0, -1], [0, -1, -1, 1], [0, -1, -1, 0], [0, -1, -1, -1],
    ]).cuda() # HxWxL
    x_conv1_coor_inbox = batch_dict['voxel_coords_inbox'].long()

    x_conv2_coor_inbox = (x_conv1_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv1_coor_inbox.shape[0])).view(-1,4)
    x_conv2_coor_inbox[:,1:] = x_conv2_coor_inbox[:,1:]//2
    x_conv2_coor_inbox = x_conv2_coor_inbox.unique(dim=0)

    x_conv3_coor_inbox = (x_conv2_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv2_coor_inbox.shape[0])).view(-1,4)
    x_conv3_coor_inbox[:,1:] = x_conv3_coor_inbox[:,1:]//2
    x_conv3_coor_inbox = x_conv3_coor_inbox.unique(dim=0)

    x_conv4_coor_inbox = (x_conv3_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv3_coor_inbox.shape[0])).view(-1,4)
    x_conv4_coor_inbox[:,1:] = x_conv4_coor_inbox[:,1:]//2
    x_conv4_coor_inbox = x_conv4_coor_inbox.unique(dim=0)

    encoded_spconv_tensor = (x_conv4_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv4_coor_inbox.shape[0])).view(-1,4)
    encoded_spconv_tensor[:,-1] = encoded_spconv_tensor[:,-1]//2
    encoded_spconv_tensor = encoded_spconv_tensor.unique(dim=0)

    batch_dict['voxel_coords_inbox_dict'] = {
                'x_input': x_conv1_coor_inbox,
                'x_conv1': x_conv1_coor_inbox,
                'x_conv2': x_conv2_coor_inbox,
                'x_conv3': x_conv3_coor_inbox,
                'x_conv4': x_conv4_coor_inbox,
                'encoded_spconv_tensor': encoded_spconv_tensor,
    }
    batch_dict.pop('voxel_coords_inbox')
    return batch_dict
    
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
            same_indices, diff_indices = get_same_indices(out.indices.long(), self.sub_indices.long())
            out.features[diff_indices] = 0

        out = self.conv2(out)
        out.features = self.bn2(out.features)
        if self.downsample is not None:
            identity = self.downsample(x)


        out.features += identity.features
        out.features = self.relu(out.features)

        return out

class VoxelBackBone8x(nn.Module):
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

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
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
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        # print("x_conv4:",x_conv4)
        # print("indices:",x_conv4.indices)
        # print("indice_dict:",x_conv4.indice_dict['x_conv4']) # (outids, indices, indice_pairs, indice_pair_num, spatial_shape)
        # for key, values in x_conv4.indice_dict.items():
        #     print("key:",key)
        #     print("outids:",x_conv4.indice_dict[key][0].shape)
        #     print("indices:",x_conv4.indice_dict[key][1].shape)
        #     print("indices:",x_conv4.indice_dict[key][1])
        #     print("indice_pair_num:",x_conv4.indice_dict[key][3])
        #     print("spatial_shape:",x_conv4.indice_dict[key][4])

        #     print("indice_pairs:",x_conv4.indice_dict[key][2].shape)
        #     print("indice_pairs max:",x_conv4.indice_dict[key][2].max())
        #     print("indice_pairs 13 sort:",torch.sort(x_conv4.indice_dict[key][2][13][0]))

        #     for i in range(x_conv4.indice_dict[key][2].shape[0]):
        #         print("out: ", i,' - ',x_conv4.indice_dict[key][2][i,1,0], ' - ' ,x_conv4.indice_dict[key][0][x_conv4.indice_dict[key][2][i,1,0]] )
        #         print("in: ", i,' - ',x_conv4.indice_dict[key][2][i,0,0], ' - ' ,x_conv4.indice_dict[key][1][x_conv4.indice_dict[key][2][i,0,0]] )

            # for i in range(x_conv4.indice_dict[key][2].shape[0]):
            #     print()
            #     print("all:",i,'  -  ',x_conv4.indice_dict[key][2][i,:,:3])
            #     if x_conv4.indice_dict[key][2][i,0,0] > 0:
            #     print(i,' - ',x_conv4.indice_dict[key][2][i,0,0], ' - ' ,x_conv4.indice_dict[key][0][x_conv4.indice_dict[key][2][i,0,0]] )
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_input': x,
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        if 'voxel_coords_inbox' in batch_dict:
            batch_dict = get_voxel_coords_inbox_dict(batch_dict)
        return batch_dict


class VoxelResBackBone8x(nn.Module):
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

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
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
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

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

        if 'voxel_coords_inbox' in batch_dict:
            batch_dict = get_voxel_coords_inbox_dict(batch_dict)
            
        return batch_dict

class VoxelResBackBone8x_sub(nn.Module):
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

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock_sub(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock_sub(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock_sub(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock_sub(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock_sub(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock_sub(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock_sub(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock_sub(128, 128, norm_fn=norm_fn, indice_key='res4'),
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

class VoxelBackBone8x_v2(nn.Module):
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

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='spconv1', conv_type='spconv'),
        ) # elodie subm -> spconv

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            # block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='spconv222', conv_type='spconv'), # elodie subm -> spconv
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'), 
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'), 
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
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
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

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

        if 'voxel_coords_inbox' in batch_dict:
            expand_index = torch.LongTensor([
                [0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 1, -1], [0, 1, 0, 1], [0, 1, 0, 0], [0, 1, 0, -1], [0, 1, -1, 1], [0, 1, -1, 0], [0, 1, -1, -1],
                [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 1, -1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 1], [0, 0, -1, 0], [0, 0, -1, -1],
                [0, -1, 1, 1], [0, -1, 1, 0], [0, -1, 1, -1], [0, -1, 0, 1], [0, -1, 0, 0], [0, -1, 0, -1], [0, -1, -1, 1], [0, -1, -1, 0], [0, -1, -1, -1],
            ]).cuda() # HxWxL
            x_conv1_coor_inbox = batch_dict['voxel_coords_inbox'].long()
            x_conv1_coor_inbox = (x_conv1_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv1_coor_inbox.shape[0])).view(-1,4).unique(dim=0)
            
            x_conv2_coor_inbox = (x_conv1_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv1_coor_inbox.shape[0])).view(-1,4).unique(dim=0)
            x_conv2_coor_inbox = (x_conv2_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv2_coor_inbox.shape[0])).view(-1,4)
            x_conv2_coor_inbox[:,1:] = x_conv2_coor_inbox[:,1:]//2
            x_conv2_coor_inbox = x_conv2_coor_inbox.unique(dim=0)

            x_conv3_coor_inbox = (x_conv2_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv2_coor_inbox.shape[0])).view(-1,4)
            x_conv3_coor_inbox[:,1:] = x_conv3_coor_inbox[:,1:]//2
            x_conv3_coor_inbox = x_conv3_coor_inbox.unique(dim=0)

            x_conv4_coor_inbox = (x_conv3_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv3_coor_inbox.shape[0])).view(-1,4)
            x_conv4_coor_inbox[:,1:] = x_conv4_coor_inbox[:,1:]//2
            x_conv4_coor_inbox = x_conv4_coor_inbox.unique(dim=0)

            encoded_spconv_tensor = (x_conv4_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv4_coor_inbox.shape[0])).view(-1,4)
            encoded_spconv_tensor[:,-1] = encoded_spconv_tensor[:,-1]//2
            encoded_spconv_tensor = encoded_spconv_tensor.unique(dim=0)

            batch_dict['voxel_coords_inbox_dict'] = {
                'x_conv1': x_conv1_coor_inbox,
                'x_conv2': x_conv2_coor_inbox,
                'x_conv3': x_conv3_coor_inbox,
                'x_conv4': x_conv4_coor_inbox,
                'encoded_spconv_tensor': encoded_spconv_tensor,
            }

            batch_dict.pop('voxel_coords_inbox')

        return batch_dict

class VoxelBackBone8x_v3(nn.Module):
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

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='spconv1', conv_type='spconv'),
        ) # elodie subm -> spconv

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='spconv22', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='spconv222', conv_type='spconv'), # elodie subm -> spconv
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='spconv33', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='spconv333', conv_type='spconv'), # elodie subm -> spconv
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='spconv44', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='spconv444', conv_type='spconv'), # elodie subm -> spconv
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
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
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

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


class VoxelBackBone8x_v4(nn.Module):
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

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='spconv1', conv_type='spconv'),
        ) # elodie subm -> spconv

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'), 
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'), 
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
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
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

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

        if 'voxel_coords_inbox' in batch_dict:
            expand_index = torch.LongTensor([
                [0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 1, -1], [0, 1, 0, 1], [0, 1, 0, 0], [0, 1, 0, -1], [0, 1, -1, 1], [0, 1, -1, 0], [0, 1, -1, -1],
                [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 1, -1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 1], [0, 0, -1, 0], [0, 0, -1, -1],
                [0, -1, 1, 1], [0, -1, 1, 0], [0, -1, 1, -1], [0, -1, 0, 1], [0, -1, 0, 0], [0, -1, 0, -1], [0, -1, -1, 1], [0, -1, -1, 0], [0, -1, -1, -1],
            ]).cuda() # HxWxL
            x_conv1_coor_inbox = batch_dict['voxel_coords_inbox'].long()
            x_conv1_coor_inbox = (x_conv1_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv1_coor_inbox.shape[0])).view(-1,4).unique(dim=0)
            
            x_conv2_coor_inbox = (x_conv1_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv1_coor_inbox.shape[0])).view(-1,4)
            x_conv2_coor_inbox[:,1:] = x_conv2_coor_inbox[:,1:]//2
            x_conv2_coor_inbox = x_conv2_coor_inbox.unique(dim=0)

            x_conv3_coor_inbox = (x_conv2_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv2_coor_inbox.shape[0])).view(-1,4)
            x_conv3_coor_inbox[:,1:] = x_conv3_coor_inbox[:,1:]//2
            x_conv3_coor_inbox = x_conv3_coor_inbox.unique(dim=0)

            x_conv4_coor_inbox = (x_conv3_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv3_coor_inbox.shape[0])).view(-1,4)
            x_conv4_coor_inbox[:,1:] = x_conv4_coor_inbox[:,1:]//2
            x_conv4_coor_inbox = x_conv4_coor_inbox.unique(dim=0)

            encoded_spconv_tensor = (x_conv4_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv4_coor_inbox.shape[0])).view(-1,4)
            encoded_spconv_tensor[:,-1] = encoded_spconv_tensor[:,-1]//2
            encoded_spconv_tensor = encoded_spconv_tensor.unique(dim=0)

            batch_dict['voxel_coords_inbox_dict'] = {
                'x_conv1': x_conv1_coor_inbox,
                'x_conv2': x_conv2_coor_inbox,
                'x_conv3': x_conv3_coor_inbox,
                'x_conv4': x_conv4_coor_inbox,
                'encoded_spconv_tensor': encoded_spconv_tensor,
            }

            batch_dict.pop('voxel_coords_inbox')

        return batch_dict

class VoxelBackBone8x_v5(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            # spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            spconv.SparseConv3d(input_channels, 16, 3, stride=1, padding=1,
                                   bias=False, indice_key='spconv1'),
            norm_fn(16),
            nn.ReLU(),
        )# elodie subm -> spconv
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'), 
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'), 
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
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
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

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

        if 'voxel_coords_inbox' in batch_dict:
            expand_index = torch.LongTensor([
                [0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 1, -1], [0, 1, 0, 1], [0, 1, 0, 0], [0, 1, 0, -1], [0, 1, -1, 1], [0, 1, -1, 0], [0, 1, -1, -1],
                [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 1, -1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 1], [0, 0, -1, 0], [0, 0, -1, -1],
                [0, -1, 1, 1], [0, -1, 1, 0], [0, -1, 1, -1], [0, -1, 0, 1], [0, -1, 0, 0], [0, -1, 0, -1], [0, -1, -1, 1], [0, -1, -1, 0], [0, -1, -1, -1],
            ]).cuda() # HxWxL
            x_conv1_coor_inbox = batch_dict['voxel_coords_inbox'].long()
            x_conv1_coor_inbox = (x_conv1_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv1_coor_inbox.shape[0])).view(-1,4).unique(dim=0)
            
            x_conv2_coor_inbox = (x_conv1_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv1_coor_inbox.shape[0])).view(-1,4)
            x_conv2_coor_inbox[:,1:] = x_conv2_coor_inbox[:,1:]//2
            x_conv2_coor_inbox = x_conv2_coor_inbox.unique(dim=0)

            x_conv3_coor_inbox = (x_conv2_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv2_coor_inbox.shape[0])).view(-1,4)
            x_conv3_coor_inbox[:,1:] = x_conv3_coor_inbox[:,1:]//2
            x_conv3_coor_inbox = x_conv3_coor_inbox.unique(dim=0)

            x_conv4_coor_inbox = (x_conv3_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv3_coor_inbox.shape[0])).view(-1,4)
            x_conv4_coor_inbox[:,1:] = x_conv4_coor_inbox[:,1:]//2
            x_conv4_coor_inbox = x_conv4_coor_inbox.unique(dim=0)

            encoded_spconv_tensor = (x_conv4_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv4_coor_inbox.shape[0])).view(-1,4)
            encoded_spconv_tensor[:,-1] = encoded_spconv_tensor[:,-1]//2
            encoded_spconv_tensor = encoded_spconv_tensor.unique(dim=0)

            batch_dict['voxel_coords_inbox_dict'] = {
                'x_conv1': x_conv1_coor_inbox,
                'x_conv2': x_conv2_coor_inbox,
                'x_conv3': x_conv3_coor_inbox,
                'x_conv4': x_conv4_coor_inbox,
                'encoded_spconv_tensor': encoded_spconv_tensor,
            }

            batch_dict.pop('voxel_coords_inbox')

        return batch_dict


class VoxelBackBone8x_v6(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            # spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            spconv.SparseConv3d(input_channels, 16, 3, stride=1, padding=1,
                                   bias=False, indice_key='spconv0'),
            norm_fn(16),
            nn.ReLU(),
        )# elodie subm -> spconv
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='spconv1', conv_type='spconv'),
        ) # elodie subm -> spconv

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'), 
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'), 
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
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
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

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

        if 'voxel_coords_inbox' in batch_dict:
            expand_index = torch.LongTensor([
                [0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 1, -1], [0, 1, 0, 1], [0, 1, 0, 0], [0, 1, 0, -1], [0, 1, -1, 1], [0, 1, -1, 0], [0, 1, -1, -1],
                [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 1, -1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 1], [0, 0, -1, 0], [0, 0, -1, -1],
                [0, -1, 1, 1], [0, -1, 1, 0], [0, -1, 1, -1], [0, -1, 0, 1], [0, -1, 0, 0], [0, -1, 0, -1], [0, -1, -1, 1], [0, -1, -1, 0], [0, -1, -1, -1],
            ]).cuda() # HxWxL
            x_conv1_coor_inbox = batch_dict['voxel_coords_inbox'].long()
            x_conv1_coor_inbox = (x_conv1_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv1_coor_inbox.shape[0])).view(-1,4).unique(dim=0)
            x_conv1_coor_inbox = (x_conv1_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv1_coor_inbox.shape[0])).view(-1,4).unique(dim=0)
            
            x_conv2_coor_inbox = (x_conv1_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv1_coor_inbox.shape[0])).view(-1,4)
            x_conv2_coor_inbox[:,1:] = x_conv2_coor_inbox[:,1:]//2
            x_conv2_coor_inbox = x_conv2_coor_inbox.unique(dim=0)

            x_conv3_coor_inbox = (x_conv2_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv2_coor_inbox.shape[0])).view(-1,4)
            x_conv3_coor_inbox[:,1:] = x_conv3_coor_inbox[:,1:]//2
            x_conv3_coor_inbox = x_conv3_coor_inbox.unique(dim=0)

            x_conv4_coor_inbox = (x_conv3_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv3_coor_inbox.shape[0])).view(-1,4)
            x_conv4_coor_inbox[:,1:] = x_conv4_coor_inbox[:,1:]//2
            x_conv4_coor_inbox = x_conv4_coor_inbox.unique(dim=0)

            encoded_spconv_tensor = (x_conv4_coor_inbox.view(1,-1).repeat(27,1) + expand_index.repeat(1,x_conv4_coor_inbox.shape[0])).view(-1,4)
            encoded_spconv_tensor[:,-1] = encoded_spconv_tensor[:,-1]//2
            encoded_spconv_tensor = encoded_spconv_tensor.unique(dim=0)

            batch_dict['voxel_coords_inbox_dict'] = {
                'x_conv1': x_conv1_coor_inbox,
                'x_conv2': x_conv2_coor_inbox,
                'x_conv3': x_conv3_coor_inbox,
                'x_conv4': x_conv4_coor_inbox,
                'encoded_spconv_tensor': encoded_spconv_tensor,
            }

            batch_dict.pop('voxel_coords_inbox')

        return batch_dict