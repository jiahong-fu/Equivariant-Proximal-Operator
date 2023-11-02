from model import common

import torch.nn as nn
from model import FCNN_har as fn

def make_model(args, parent=False):
    return EDSR_plus(args)

class EDSR_plus(nn.Module):
    def __init__(self, args):
        super(EDSR_plus, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = int(args.kernel_size)
        iniScale = args.ini_scale
        scale = args.scale[0]
        act = nn.ReLU(True)
        inP = kernel_size
        tranNum = args.tranNum
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        Smooth = False
        m_head =  [fn.Fconv_PCA(kernel_size,args.n_colors,n_feats,tranNum,inP=inP,padding=(kernel_size-1)//2, ifIni=1, Smooth = Smooth, iniScale = iniScale)]
        

        # define body module
        m_body = [
            fn.ResBlock(
                fn.Fconv_PCA, n_feats, kernel_size,tranNum = tranNum, inP = inP,  act=act, res_scale=args.res_scale, Smooth = Smooth, iniScale = iniScale
            ) for _ in range(n_resblocks)
        ]
#        m_body.append(fn.GroupFusion(n_feats, tranNum))

        n_feats = n_feats
        m_tail = [
                fn.Fconv_PCA(kernel_size,n_feats,args.n_colors,tranNum,inP=inP,padding=(kernel_size-1)//2,  Smooth = Smooth, iniScale = iniScale),
                fn.Fconv_PCA_out(kernel_size,args.n_colors,args.n_colors,tranNum,inP=inP,padding=(kernel_size-1)//2,  Smooth = Smooth, iniScale = iniScale)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
#        x = self.sub_mean(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
#        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

