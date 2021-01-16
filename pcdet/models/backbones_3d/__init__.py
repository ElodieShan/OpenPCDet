from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x, VoxelBackBone8x_v2, VoxelBackBone8x_v3, VoxelResBackBone8x_sub
from .spconv_backbone_teacher import VoxelResBackBone8x_SA, VoxelResBackBone8x_ATTEN, VoxelBackBone8x_ATTEN, VoxelBackBone8x_ATTEN_v2, VoxelBackBone8x_ATTEN_v3, SA_VoxelBackBone8x
from .spconv_unet import UNetV2

__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'VoxelBackBone8x_v2': VoxelBackBone8x_v2,
    'VoxelBackBone8x_v3': VoxelBackBone8x_v3,
    'UNetV2': UNetV2,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    'VoxelResBackBone8x': VoxelResBackBone8x,
    'VoxelResBackBone8x_sub': VoxelResBackBone8x_sub,
    'VoxelResBackBone8x_SA' : VoxelResBackBone8x_SA,
    'VoxelBackBone8x_ATTEN': VoxelBackBone8x_ATTEN,
    'VoxelBackBone8x_ATTEN_v2': VoxelBackBone8x_ATTEN_v2,
    'VoxelBackBone8x_ATTEN_v3': VoxelBackBone8x_ATTEN_v3,
    'SA_VoxelBackBone8x': SA_VoxelBackBone8x,
}
