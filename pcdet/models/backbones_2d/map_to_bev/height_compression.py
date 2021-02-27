import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']

        if 'sub_branch' in batch_dict: #elodie
            sub_encoded_spconv_tensor = batch_dict['sub_branch']['encoded_spconv_tensor']
            sub_spatial_features = sub_encoded_spconv_tensor.dense()
            N, C, D, H, W = sub_spatial_features.shape
            sub_spatial_features = sub_spatial_features.view(N, C * D, H, W)
            batch_dict['sub_branch']['spatial_features'] = sub_spatial_features
            batch_dict['sub_branch']['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']

        return batch_dict
