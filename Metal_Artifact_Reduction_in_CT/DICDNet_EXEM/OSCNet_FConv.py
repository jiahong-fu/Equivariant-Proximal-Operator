# IJCAI 2022
# Adaptive Convolutional Dictionary Network for CT Metal Artifact Reduction

from email.policy import strict
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as  F
from torch.autograd import Variable
import scipy.io as io
import math
from PIL import Image
import FCNN_plus as fcnn

# Initialize the common dictionary D with a simple Gaussian Kernel
Dini = io.loadmat('utils/init_kernel_dir.mat') ['C9'] # 3*64*9*9
Dini= Dini[0:1, 0:32, :, :]

#### dict initialization with F_conv
###################################

# Filtering on the XLI for initializing P^(0)  and X^(0), refer to Sec 1.2 in supplementary material (SM)
filter = torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) / 9
filter = filter.unsqueeze(dim=0).unsqueeze(dim=0)
class OSCNet(nn.Module):
    def __init__(self, args):
        super(OSCNet, self).__init__()
        self.T  = args.T                                            # Stage number T includes the initialization process
        self.iters = self.T -1                                      # not include the initialization process
        # self.d = args.d                                             # the number d of kernel in the dictionary D
        # self.N = args.N                                             # the channel dimension N of the feature map M
        self.Np = args.Np                                           # the channel expansion dimension Np, refer to Sec 1.1 in supplementary material (SM)
        self.batch_size = args.batchSize

        ################# Dic init with normal
        Dic0 = torch.FloatTensor(Dini)
        Dic1 = torch.FloatTensor(Dini)
        self.F_filter0 = nn.Parameter(Dic0, requires_grad=True)
        self.F_filter = nn.Parameter(Dic1, requires_grad=True)
        #############################

        # Stepsize
        self.etaM = torch.Tensor([args.etaM])                         # initialization
        self.etaX = torch.Tensor([args.etaX])                         # initialization
        # self.etaK = torch.Tensor([5])                                 # initialization
        self.etaM_stage = self.make_eta(self.iters, self.etaM)                # learnable
        self.etaX_stage = self.make_eta(self.T, self.etaX)
        # self.eta3_T = self.make_eta(self.T, self.etaK)
        # Dic = torch.FloatTensor(Dini)
        # self.D = nn.Parameter(data=Dic[:,:self.d,:,:], requires_grad=True)

        # proxNet
        self.proxNet_X_stage = self.make_Xnet(self.T, args)
        self.proxNet_M_stage = self.make_Mnet(self.T, args)
        # self.proxNet_K_T = self.make_Knet(self.T, args)
        self.proxNet_X_init = Xnet(args)                               # using in initialization
        self.proxNet_X_last_layer = Xnet(args)                       # fine-tune at the last


        self.tau_const = torch.Tensor([1])
        self.tau = nn.Parameter(self.tau_const, requires_grad=True)  # for sparse feature map
        # self.tau_extend = nn.Parameter(self.tau_const, requires_grad=True)

        # filter for initializing X and P
        self.Cp_const = filter.expand(self.Np, 1, -1, -1)  # size: self.num_Z*1*3*3
        self.Cp = nn.Parameter(self.Cp_const, requires_grad=True)

        # For initializing M^(0), refer to Sec 1.2 in SM
        # self.etaX_nonK = self.make_eta(2, self.etaX)
        # self.etaM_nonK = self.make_eta(1, self.etaM)
        # self.D0 = nn.Parameter(data=Dic[:, :self.N, :, :], requires_grad=True)  # used in initialization process
        # self.proxNet_X_0 = Xnet(args)  # used in initialization process
        # self.proxNet_M_init = self.make_Mnet(2, args)
        # self.proxNet_X_init = self.make_Xnet(2, args)
        # convert = torch.eye(self.Np, self.Np)
        # big_convert = convert.unsqueeze(dim=2).unsqueeze(dim=3)# self.N*self.N*1*1
        # self.convert_conv_layer1 = nn.Parameter(data=big_convert, requires_grad=True)
        # self.convert_conv_layer2 = nn.Parameter(data=big_convert, requires_grad=True)

    def make_Xnet(self, iters, args):
        layers = []
        for i in range(iters):
            layers.append(Xnet(args))
        return nn.Sequential(*layers)


    def make_Mnet(self, iters, args):                           # Mnet channels No. N + num_ZM
        layers = []
        for i in range(iters):
            layers.append(Mnet(args))
        return nn.Sequential(*layers)

    # def make_Knet(self, iters, args):
    #     layers = []
    #     for i in range(iters):
    #         layers.append(Knet(args))
    #     return nn.Sequential(*layers)

    def make_eta(self, iters, const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters, -1)
        eta = nn.Parameter(data=const_f, requires_grad=True)
        return eta

    def GetBasis_PCA(self, sizeP, tranNum=8, inP=None, Smooth = True):
        if inP==None:
            inP = sizeP
        inX, inY, Mask = self.MaskC(sizeP)
        X0 = np.expand_dims(inX,2)
        Y0 = np.expand_dims(inY,2)
        Mask = np.expand_dims(Mask,2)
        theta = np.arange(tranNum)/tranNum*2*np.pi
        theta = np.expand_dims(np.expand_dims(theta,axis=0),axis=0)
    #    theta = torch.FloatTensor(theta)
        X = np.cos(theta)*X0-np.sin(theta)*Y0
        Y = np.cos(theta)*Y0+np.sin(theta)*X0
    #    X = X.unsqueeze(3).unsqueeze(4)
        X = np.expand_dims(np.expand_dims(X,3),4)
        Y = np.expand_dims(np.expand_dims(Y,3),4)
        v = np.pi/inP*(inP-1)
        p = inP/2
        
        k = np.reshape(np.arange(inP),[1,1,1,inP,1])
        l = np.reshape(np.arange(inP),[1,1,1,1,inP])
        
        
        BasisC = np.cos((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y)
        BasisS = np.sin((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y)
        
        BasisC = np.reshape(BasisC,[sizeP, sizeP, tranNum, inP*inP])*np.expand_dims(Mask,3)
        BasisS = np.reshape(BasisS,[sizeP, sizeP, tranNum, inP*inP])*np.expand_dims(Mask,3)

        BasisC = np.reshape(BasisC,[sizeP*sizeP*tranNum, inP*inP])
        BasisS = np.reshape(BasisS,[sizeP*sizeP*tranNum, inP*inP])

        BasisR = np.concatenate((BasisC, BasisS), axis = 1)
        
        U,S,VT = np.linalg.svd(np.matmul(BasisR.T,BasisR))

        Rank   = np.sum(S>0.0001)
        BasisR = np.matmul(np.matmul(BasisR,U[:,:Rank]),np.diag(1/np.sqrt(S[:Rank]+0.0000000001))) 
        BasisR = np.reshape(BasisR,[sizeP, sizeP, tranNum, Rank])
        
        temp = np.reshape(BasisR, [sizeP*sizeP, tranNum, Rank])
        var = (np.std(np.sum(temp, axis = 0)**2, axis=0)+np.std(np.sum(temp**2*sizeP*sizeP, axis = 0),axis = 0))/np.mean(np.sum(temp, axis = 0)**2+np.sum(temp**2*sizeP*sizeP, axis = 0),axis = 0)
        Trod = 1
        Ind = var<Trod
        Rank = np.sum(Ind)
        Weight = 1/np.maximum(var, 0.04)/25
        if Smooth:
            BasisR = np.expand_dims(np.expand_dims(np.expand_dims(Weight,0),0),0)*BasisR

        return torch.FloatTensor(BasisR), Rank, Weight

    def Getini_reg(self, nNum, inNum, outNum,expand, weight = 1): 
        A = (np.random.rand(outNum,inNum,expand,nNum)-0.5)*2*2.4495/np.sqrt((inNum)*nNum)*np.expand_dims(np.expand_dims(np.expand_dims(weight, axis = 0),axis = 0),axis = 0)
        return torch.FloatTensor(A)

    def MaskC(self, SizeP):
            p = (SizeP-1)/2
            x = np.arange(-p,p+1)/p
            X,Y  = np.meshgrid(x,x)
            C    =X**2+Y**2
            
            Mask = np.ones([SizeP,SizeP])
    #        Mask[C>(1+1/(4*p))**2]=0
            Mask = np.exp(-np.maximum(C-1,0)/0.2)
            
            return X, Y, Mask

    def forward(self, Xma, XLI, Mask):  # Mask: non-metal region
    ## First using non-K version to initialize M and X, refer to Sec 1.2 in SM
        input = Xma
        b, h, w = input.size()[0], input.size()[2], input.size()[3]
        ListX =[]
        ListA =[]
        ######## calculating dict ###############
        # tempW0 = torch.einsum('ijok,mnak->monaij', self.Basis0, self.weights0)
        # F_filter0 = tempW0.reshape([4*8, 1, 9, 9 ]).permute(1, 0, 2, 3)
        # tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)
        # F_filter = tempW.reshape([4*8, 1, 9, 9 ]).permute(1, 0, 2, 3)
        # print("self.weight[0][0]:", self.weights[0][0])
        #########################################
        # ListX_nonK = []
        # ListA_nonK = []
        # initialize P0 and X0
        P00 = F.conv2d(XLI, self.Cp, stride=1, padding=1)
        input_ini = torch.cat((XLI, P00), dim=1)
        XP_ini = self.proxNet_X_init(input_ini)
        X0 = XP_ini[:, :1, :, :]
        P0 = XP_ini[:, 1:, :, :]

        # updating M0--->M1
        ES = Mask*(input - X0)
        EDM = F.relu(ES - self.tau)                                            #for sparse rain layer
        # 这里相当M^{0}初始为0了，所以没有C \otimes M^{0}
        ########## RCD专门给了一个F_filtre来做M的初始化，这里我们暂用一样的，因为ACD是一样
        GM = F.conv_transpose2d(EDM, self.F_filter0/10, stride=1, padding=4)   # /10 for controlling the updating speed
        # print("GM.shape", GM.shape)
        #######################################
        M = self.proxNet_M_stage[0](GM)
        DM = F.conv2d(M, self.F_filter/10, stride =1, padding = 4)
        # Updating X0-->X1
        EB = input - DM
        GX = X0-EB  # X0 - input + DM
        X1 = X0-self.etaX_stage[0,:]/10*Mask*GX
        input_dual = torch.cat((X1, P0), dim=1)
        out_dual = self.proxNet_X_stage[0](input_dual)
        X = out_dual[:,:1,:,:]
        P = out_dual[:,1:,:,:]
        ListX.append(X)
        ListA.append(DM)

        for i in range(self.iters):
            # M-net
            ES = input - X
            EDM = DM - ES
            GM = F.conv_transpose2d(Mask*EDM, self.F_filter/10, stride =1, padding = 4)
            input_new = M - self.etaM_stage[i, :] / 10 * GM 
            M = self.proxNet_M_stage[i+1](input_new)

            # X-net
            DM = F.conv2d(M, self.F_filter/10, stride =1, padding = 4)
            EX = input - DM
            GX = X - EX
            x_dual = X - self.etaX_stage[i+1, :] / 10 * Mask * GX
            input_dual = torch.cat((x_dual, P), dim=1)
            out_dual = self.proxNet_X_stage[i+1](input_dual)
            X = out_dual[:, :1, :, :]
            P = out_dual[:,1:,:,:]

            # Maybe adjust layers
            # M1 = F.conv2d(F.relu(M),self.convert_conv_layer1, stride=1, padding=0)
            # M = F.conv2d(M1, self.convert_conv_layer2, stride=1, padding=0)

            ListX.append(X)
            ListA.append(DM)
        XP_adjust = self.proxNet_X_last_layer(out_dual)
        X= XP_adjust[:, :1, :, :]
        ListX.append(X)
        return X0, ListX, ListA

        
        


# porxNet_M Fconv
class Mnet(nn.Module):
    def __init__(self, args):
        super(Mnet, self).__init__()
        self.channels = args.Np
        self.L = args.num_res
        # self.layer = self.make_resblock(self.L)
        self.tau0 = torch.Tensor([args.Mtau])
        self.tau_const = self.tau0.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).expand(-1,self.channels,-1,-1)
        self.tau = nn.Parameter(self.tau_const, requires_grad=True)  # for sparse rain map
        
        kernel_size=5
        n_feats=4
        tranNum=8
        inP=kernel_size
        Smooth=False
        act=nn.ReLU(True)
        m_body = [
        fcnn.ResBlock(
            fcnn.Fconv_PCA, n_feats, kernel_size, tranNum, inP, bn=True, act=act, res_scale=1.0, Smooth=Smooth, iniScale=1.0
        ) for _ in range(self.L)
        ]
        self.body = nn.Sequential(*m_body)
        # self.resm = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1))

    def forward(self, input):
        # above feather maps of M had been manipulated in groups conv
        res = self.body(input)
        # m = F.relu(input + 0.01 * self.resm(res))
        # m = F.relu(m - self.tau)
        m = F.relu(res - self.tau)
        return m


# ProxNet_X FConv
class Xnet(nn.Module):
    def __init__(self, args):
        super(Xnet, self).__init__()
        self.channels = args.Np + 1
        self.L = args.num_res

        kernel_size = 5
        n_feats = 4
        tranNum = 8
        inP = kernel_size
        Smooth = False
        act = nn.ReLU(True)
        self.head_in = nn.Sequential(fcnn.Fconv_PCA(kernel_size, self.channels, n_feats, tranNum, inP, padding=(kernel_size-1)//2, ifIni=1, Smooth=Smooth, iniScale=1.0))
        m_body = [
            fcnn.ResBlock(
                fcnn.Fconv_PCA, n_feats, kernel_size, tranNum=tranNum, inP=inP, bn=True, act=act, res_scale=0.1, Smooth=Smooth, iniScale=1.0
            ) for _ in range(self.L)
        ]
        self.body = nn.Sequential(*m_body)
        self.out = nn.Sequential(fcnn.Fconv_PCA_out(kernel_size, n_feats, self.channels, tranNum, inP=inP, padding=(kernel_size-1)//2, ifIni=0, Smooth=Smooth, iniScale=1.0))
        # self.resx = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1))

    def forward(self, input):
        x_f = self.head_in(input)
        res = self.body(x_f)
        res_x = self.out(res)
        # x = F.relu(input + 0.01 * self.resx(res_x))
        x = F.relu(input + 0.1 * res_x)
        return x

# class FConv_filter(object):
#     def __init__(self, I=1):
#         self.I = I

#     def __call__(self, I = 1):
#         Basis, Rank, weight = GetBasis_PCA(sizeP=9,tranNum=8,inP=9, Smooth = True)
#         iniw = Getini_reg(Basis.size(3), inNum=1, outNum=4, expand=1, weight=weight) #iniScale
#         Basis = Basis0.cuda()
#         weights = nn.Parameter(iniw, requires_grad=True).cuda()
#         tempW = torch.einsum('ijok,mnak->monaij', Basis, weights)
#         Num = 8  # tranNum//expand
#         tempWList = [torch.cat([tempW[:,i*Num:(i+1)*Num,:,-i:,:,:],tempW0[:,i*Num:(i+1)*Num,:,:-i,:,:]], dim = 3) for i in range(1)]
#         tempW = torch.cat(tempWList, dim = 1)
#         self.F_filter = tempW.reshape([4*8, 1, 9, 9 ]).permute(1, 0, 2, 3)
#         return self.F_filter

# def GetBasis_PCA(sizeP, tranNum=8, inP=None, Smooth = True):
#     if inP==None:
#         inP = sizeP
#     inX, inY, Mask = MaskC(sizeP)
#     X0 = np.expand_dims(inX,2)
#     Y0 = np.expand_dims(inY,2)
#     Mask = np.expand_dims(Mask,2)
#     theta = np.arange(tranNum)/tranNum*2*np.pi
#     theta = np.expand_dims(np.expand_dims(theta,axis=0),axis=0)
# #    theta = torch.FloatTensor(theta)
#     X = np.cos(theta)*X0-np.sin(theta)*Y0
#     Y = np.cos(theta)*Y0+np.sin(theta)*X0
# #    X = X.unsqueeze(3).unsqueeze(4)
#     X = np.expand_dims(np.expand_dims(X,3),4)
#     Y = np.expand_dims(np.expand_dims(Y,3),4)
#     v = np.pi/inP*(inP-1)
#     p = inP/2
    
#     k = np.reshape(np.arange(inP),[1,1,1,inP,1])
#     l = np.reshape(np.arange(inP),[1,1,1,1,inP])
    
    
#     BasisC = np.cos((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y)
#     BasisS = np.sin((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y)
    
#     BasisC = np.reshape(BasisC,[sizeP, sizeP, tranNum, inP*inP])*np.expand_dims(Mask,3)
#     BasisS = np.reshape(BasisS,[sizeP, sizeP, tranNum, inP*inP])*np.expand_dims(Mask,3)

#     BasisC = np.reshape(BasisC,[sizeP*sizeP*tranNum, inP*inP])
#     BasisS = np.reshape(BasisS,[sizeP*sizeP*tranNum, inP*inP])

#     BasisR = np.concatenate((BasisC, BasisS), axis = 1)
    
#     U,S,VT = np.linalg.svd(np.matmul(BasisR.T,BasisR))

#     Rank   = np.sum(S>0.0001)
#     BasisR = np.matmul(np.matmul(BasisR,U[:,:Rank]),np.diag(1/np.sqrt(S[:Rank]+0.0000000001))) 
#     BasisR = np.reshape(BasisR,[sizeP, sizeP, tranNum, Rank])
    
#     temp = np.reshape(BasisR, [sizeP*sizeP, tranNum, Rank])
#     var = (np.std(np.sum(temp, axis = 0)**2, axis=0)+np.std(np.sum(temp**2*sizeP*sizeP, axis = 0),axis = 0))/np.mean(np.sum(temp, axis = 0)**2+np.sum(temp**2*sizeP*sizeP, axis = 0),axis = 0)
#     Trod = 1
#     Ind = var<Trod
#     Rank = np.sum(Ind)
#     Weight = 1/np.maximum(var, 0.04)/25
#     if Smooth:
#         BasisR = np.expand_dims(np.expand_dims(np.expand_dims(Weight,0),0),0)*BasisR

#     return torch.FloatTensor(BasisR), Rank, Weight

# def Getini_reg(nNum, inNum, outNum,expand, weight = 1): 
#     A = (np.random.rand(outNum,inNum,expand,nNum)-0.5)*2*2.4495/np.sqrt((inNum)*nNum)*np.expand_dims(np.expand_dims(np.expand_dims(weight, axis = 0),axis = 0),axis = 0)
#     return torch.FloatTensor(A)

# def MaskC(SizeP):
#     p = (SizeP-1)/2
#     x = np.arange(-p,p+1)/p
#     X,Y  = np.meshgrid(x,x)
#     C    =X**2+Y**2
    
#     Mask = np.ones([SizeP,SizeP])
# #        Mask[C>(1+1/(4*p))**2]=0
#     Mask = np.exp(-np.maximum(C-1,0)/0.2)
    
#     return X, Y, Mask
