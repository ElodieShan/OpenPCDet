import spconv
import torch.nn as nn
import torch
import numpy as np

class PosEncoding(nn.Module):
    def __init__(self, d_word_vec, max_seq_len= 211200):
        super(PosEncoding, self).__init__()
        pos_enc = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / d_word_vec) for j in range(d_word_vec)]
            for pos in range(max_seq_len)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
        pad_row = np.zeros([1, d_word_vec])
        pos_enc = np.concatenate([pad_row, pos_enc]).astype(np.float32)
        self.max_seq_len = max_seq_len
        # additional single row for PAD idx
        self.pos_enc = nn.Embedding(max_seq_len + 1, d_word_vec)
        # fix positional encoding: exclude weight from grad computation
        self.pos_enc.weight = nn.Parameter(torch.from_numpy(pos_enc), requires_grad=False)

    def forward(self, input_pos):
        # max_len = torch.max(input_len)
        # tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # input_pos = tensor([list(range(1, len+1)) + [0]*(max_len-len) for len in input_len])

        return self.pos_enc(input_pos)

    # def forward(self, input_len):
    #     max_len = torch.max(input_len)
    #     tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
    #     input_pos = tensor([list(range(1, len+1)) + [0]*(max_len-len) for len in input_len])

    #     return self.pos_enc(input_pos)

class MappingBlock(nn.Module):
    def __init__(self, input_channels,  output_channels, mid_channels=None, kernel_size=1, stride=1):
        super(MappingBlock, self).__init__()
        if mid_channels is None:
            mid_channels = output_channels

        self.conv_map = nn.Sequential(
                            nn.Conv1d(
                                input_channels, mid_channels,
                                kernel_size=kernel_size, stride=stride
                            ),
                            nn.BatchNorm1d(mid_channels, eps=1e-3, momentum=0.01),
                            nn.ReLU(),
                            nn.Conv1d(
                                mid_channels, output_channels,
                                kernel_size=1, 
                            ),
                    )

    def forward(self, x):
        out = self.conv_map(x)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim=32, num_heads=3, max_seq_len=5000, pos_shape=None):
        super(AttentionBlock, self).__init__()
        # def __init__(self, input_channels, output_channels, kernel_size=1, stride=1, embed_dim=64, num_heads=3):

        # self.conv_map = nn.Sequential(
        #                     nn.Conv1d(
        #                         input_channels, output_channels,
        #                         kernel_size=kernel_size, stride=stride
        #                     ),
        #                     nn.BatchNorm1d(output_channels, eps=1e-3, momentum=0.01),
        #                     nn.ReLU(),
        #                     nn.Conv1d(
        #                         output_channels, output_channels,
        #                         kernel_size=1, 
        #                     ),
        #             )
        self.indices_pos_shape = [pos_shape[0],pos_shape[1]*pos_shape[0]]
        self.pos_emb = PosEncoding(embed_dim, max_seq_len)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def indices_to_pos(self, indices):
        return indices[:,1] + indices[:,2]*self.indices_pos_shape[0] + indices[:,3]*self.indices_pos_shape[1] + 1

    def forward(self, ori_feature_dict, sub_feature_dict, batch_size):
        # out = self.conv_map(teacher_feature)
        # out = out.permute(2,0,1)
        # self.pos_emb()
        # key=value
        # batch_cnt = torch.LongTensor([(indices[:,0]==i).sum() for i in range(batch_size)]).cuda()
        ori_feature = ori_feature_dict.features
        ori_indices = ori_feature_dict.indices
        ori_feature += self.pos_emb(self.indices_to_pos(ori_indices).long())
        query_feature = sub_feature_dict.features
        query_indices = sub_feature_dict.indices
        query_feature += self.pos_emb(self.indices_to_pos(query_indices).long())

        feature_dim = ori_feature.shape[-1]

        # sub_query_indices = indices[sub_query_index]
        # batch_cnt_query = torch.LongTensor([(sub_query_indices[:,0]==i).sum() for i in range(batch_size)])
        for i in range(batch_size):
            batch_mask = ori_indices[:,0]==i
            batch_query_mask = query_indices[:,0]==i

            # print("indices:",indices.shape)
            # print("sub_query_index:",sub_query_index.shape)
            # print("batch_mask:",batch_mask.shape)

            # print("query_feature[batch_query_mask]:",query_feature[batch_query_mask].shape)
            # print(query_feature[batch_query_mask].view(1,batch_query_mask.sum(),feature_dim))
            query_feature[batch_query_mask] = self.multihead_attn(
                query_feature.clone()[batch_query_mask].view(1,batch_query_mask.sum(),feature_dim).permute(1,0,2), 
                ori_feature[batch_mask].view(1,batch_mask.sum(),feature_dim).permute(1,0,2),
                ori_feature[batch_mask].view(1,batch_mask.sum(),feature_dim).permute(1,0,2))[0].permute(1,0,2).view(-1,feature_dim)
        # print("query_feature af:",query_feature)
        # print("sub_feature_dict af:",sub_feature_dict.features)

        # print("attn_output:",attn_output)
    
        return query_feature