import spconv
import torch.nn as nn
import torch

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

        # additional single row for PAD idx
        self.pos_enc = nn.Embedding(max_seq_len + 1, d_word_vec)
        # fix positional encoding: exclude weight from grad computation
        self.pos_enc.weight = nn.Parameter(torch.from_numpy(pos_enc), requires_grad=False)

    def forward(self, input_len):
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        input_pos = tensor([list(range(1, len+1)) + [0]*(max_len-len) for len in input_len])

        return self.pos_enc(input_pos)

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
    def __init__(self, embed_dim=32, num_heads=3):
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
        # max_seq_len = 5000
        # self.pos_emb = PosEncoding(max_seq_len * 10, embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, teacher_feature, student_feature):
        # out = self.conv_map(teacher_feature)
        # out = out.permute(2,0,1)
        # self.pos_emb()
        attn_output, attn_output_weights = self.multihead_attn(student_feature, teacher_feature, teacher_feature)
        # print("attn_output:",attn_output)
    
        return attn_output