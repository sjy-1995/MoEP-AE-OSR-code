# coding:utf-8
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import distributions_utils_new as ut
import numpy as np
import torchvision.models as models
from torch.distributions import kl_divergence

# for vision transformer
# from swin_transformer_pytorch import SwinTransformer   # the simpler file
from swin_transformer import SwinTransformer   # the more complex file


class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class Extract_ResNet18(nn.Module):
    def __init__(self, model):
        super(Extract_ResNet18, self).__init__()
        # 取掉model的后两层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        # print(x.shape)
        x = self.resnet_layer(x)
        # print(x.shape)
        return x


class openSetClassifier(nn.Module):

    def __init__(self, num_classes=20, num_channels=3, im_size=224, init_weights=True, dropout=0.2, num_Blocks=[2, 2, 2, 2], nc=3, **kwargs):

        super(openSetClassifier, self).__init__()
        self.num_classes = num_classes
        self.in_planes = 512
        self.prob_latent = PROB_LATENT(num_classes, num_channels, im_size)
        self.n = 256
        self.k = 4

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)

        # ###################################### for encoder ###################################################
        # resnet = models.resnet50(pretrained=True)
        # modules = list(resnet.children())[:-1]   # delete the last fc layer
        # self.resnet50 = nn.Sequential(*modules)

        # try to adopt the vision transformer as the encoder
        # # swin_t
        # self.swin_vit = SwinTransformer(
        #     hidden_dim=96,
        #     layers=(2, 2, 6, 2),
        #     heads=(3, 6, 12, 24),
        #     channels=3,
        #     num_classes=3,
        #     head_dim=32,
        #     window_size=7,
        #     downscaling_factors=(4, 2, 2, 2),
        #     relative_pos_embedding=True
        # )
        # # swin_l
        # self.swin_vit = SwinTransformer(
        #     hidden_dim=192,
        #     layers=(2, 2, 18, 2),
        #     heads=(6, 12, 24, 48),
        #     channels=3,
        #     num_classes=1000,   # use the pretrained model trained on ImageNet
        #     head_dim=32,
        #     window_size=7,
        #     downscaling_factors=(4, 2, 2, 2),
        #     relative_pos_embedding=True
        # )
        # # swin_b
        # self.swin_vit = SwinTransformer(
        #     hidden_dim=128,
        #     layers=(2, 2, 18, 2),
        #     heads=(4, 8, 16, 32),
        #     channels=3,
        #     num_classes=1000,   # use the pretrained model trained on ImageNet
        #     head_dim=32,
        #     window_size=7,
        #     downscaling_factors=(4, 2, 2, 2),
        #     relative_pos_embedding=True
        # )
        # swin_b for more complex model file
        self.swin_vit = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False
                 # use_checkpoint=True
        )

        # some fc layers
        # self.fc1 = nn.Linear(1536, 768)   # swin_l
        self.fc1 = nn.Linear(1024, 768)   # swin_b
        self.bn1 = nn.BatchNorm1d(768, momentum=0.01)
        self.fc2 = nn.Linear(768, 768)
        self.bn2 = nn.BatchNorm1d(768, momentum=0.01)

        # ###################################### for decoder ####################################################
        # some fc layers
        self.fc4 = nn.Linear(256, 768)
        self.bn4 = nn.BatchNorm1d(768)
        self.fc5 = nn.Linear(768, 64 * 4 * 4)
        self.bn5 = nn.BatchNorm1d(64 * 4 * 4)

        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans9 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans10 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans11 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid(),   # in [0, 1]
            # nn.Tanh(),   # in [-1, 1]
        )

        # ############################### for attention module before the classifier ##############################
        self.sigmap_fc = nn.Sequential(
            nn.Linear(self.k * self.n, self.k),
            nn.ReLU(inplace=True),
            nn.Linear(self.k, self.k),
            nn.ReLU(inplace=True),
        )
        self.p_fc = nn.Sequential(
            nn.Linear(self.k * self.n, self.k),
            nn.ReLU(inplace=True),
            nn.Linear(self.k, self.k),
            nn.ReLU(inplace=True),
        )
        self.sigmap_p_fc = nn.Sequential(
            nn.Linear(self.k, self.k),
            nn.Softmax(dim=1),
        )

        # ###################################### for classifier ###################################################
        # self.c1 = nn.Linear(512 * 4, 4096)
        # self.c1_mu = nn.Linear(512 * 4, 4096)
        # self.c2 = nn.Linear(4096, self.num_classes)
        # self.c2_mu = nn.Linear(4096, self.num_classes)
        self.classifier = nn.Linear(256 * 4, self.num_classes)
        # self.classifier = nn.Linear(256, self.num_classes)

    def forward(self, x, y_de):

        # encoder
        batch_size = len(x)
        # model1 = torch.load('imagenet21k+imagenet2012_ViT-B_16-224.pth')
        # print(model1)

        # for k in self.swin_vit.state_dict():
        #     print(k)
        # print('_______________________________________________________')
        # for k in torch.load('swin_large_patch4_window7_224_22k.pth')['model']:
        #     print(k)

        # model_dict1 = self.swin_vit.state_dict()
        # # try:
        #     # model_2_ = torch.load('../improved-wgan-pytorch-master/networks/weights/{}/{}_{}_try3_F_and_C_unifiedAccuracy.pth'.format(args.dataset, args.dataset, args.trial))
        # # model_2 = torch.load('swin_large_patch4_window7_224_22k.pth')   # swin_l
        # model_2 = torch.load('swin_base_patch4_window7_224_22k.pth')   # swin_b
        # # except:
        # #     # model_2_ = torch.load('../../improved-wgan-pytorch-master/networks/weights/{}/{}_{}_try3_F_and_C_unifiedAccuracy.pth'.format(args.dataset, args.dataset, args.trial))
        # #     model_2_ = torch.load('networks/weights/{}/{}_{}_try3_F_and_C_unifiedAccuracy.pth'.format(args.dataset, args.dataset, args.trial))
        # # model_2 = Net1(model_2_)
        # # model_dict2 = model_2.state_dict()
        # model_dict2 = model_2['model']
        # model_list1 = list(model_dict1.keys())
        # model_list2 = list(model_dict2.keys())
        # len1 = len(model_list1)
        # len2 = len(model_list2)
        # minlen = min(len1, len2)
        # for n in range(minlen):
        #     # print(model_dict1[model_list1[n]])
        #     if model_dict1[model_list1[n]].shape != model_dict2[model_list2[n]].shape:
        #         print('no equal!!!')
        #         continue
        #     model_dict1[model_list1[n]] = model_dict2[model_list2[n]]
        # self.swin_vit.load_state_dict(model_dict1)

        # self.swin_vit.load_state_dict(torch.load('swin_large_patch4_window7_224_22k.pth'))   # use the pretrained model trained on ImageNet
        # self.swin_vit.load_state_dict(torch.load('swin_base_patch4_window7_224_22k.pth')['model'], strict=False)   # use the pretrained model trained on ImageNet
        # self.swin_vit = torch.load('swin_base_patch4_window7_224_22k.pth')   # use the pretrained model trained on ImageNet

        # # for provide grad under the condition that uses checkpoint.checkpoint
        # x = x + torch.zeros(1, dtype=x.dtype, device=x.device, requires_grad=True)

        x1 = self.swin_vit(x)
        # print(x1.shape)
        x1 = x1.view(batch_size, -1)   # b, 1536
        # print(x1.shape)
        x1 = self.relu(self.bn1(self.fc1(x1)))   # b, 768
        # print(x1.shape)
        x1 = self.relu(self.bn2(self.fc2(x1)))   # b, 768
        # print(x1.shape)

        # sampler
        samples_latent, mu, sigmap, p = self.prob_latent(x1)  # (b, n), (b, n, k)

        # decoder
        # x_re = self.relu(self.bn4(self.fc4(samples_latent)))   # (b, 768)
        # x_re = self.relu(self.bn5(self.fc5(x_re)))   # (b, 64*4*4(1024))
        # x_re = x_re.view(-1, 64, 4, 4)   # (b, 64, 4, 4)
        # x_re = self.convTrans6(x_re)
        # x_re = self.convTrans7(x_re)
        # x_re = self.convTrans8(x_re)
        # x_re = F.interpolate(x_re, size=(224, 224), mode='bilinear')   # (b, 3, 224, 224)

        # change the decoder
        x_re = self.relu(self.bn4(self.fc4(samples_latent)))   # (b, 768)
        x_re = self.relu(self.bn5(self.fc5(x_re)))   # (b, 64*4*4(1024))
        x_re = x_re.view(-1, 64, 4, 4)   # (b, 64, 4, 4)
        x_re = self.convTrans6(x_re)   # (b, 64, 8, 8)
        x_re = self.convTrans7(x_re)   # (b, 64, 16, 16)
        x_re = self.convTrans8(x_re)   # (b, 64, 32, 32)
        x_re = self.convTrans9(x_re)   # (b, 64, 64, 64)
        x_re = self.convTrans10(x_re)   # (b, 32, 128, 128)
        x_re = self.convTrans11(x_re)   # (b, 3, 256, 256)
        x_re = F.interpolate(x_re, size=(224, 224), mode='bilinear')   # (b, 3, 224, 224)

        # reconstruction loss
        # ############# to use MSE loss or BCE loss ?????? ################
        reconstruction_function = nn.MSELoss()
        reconstruction_function.size_average = False
        rec = reconstruction_function(x_re, x)

        # classification
        mu_ = mu.permute(0, 2, 1).contiguous().view(batch_size, 256 * self.k)  # (b, n*k)
        sigmap_ = sigmap.permute(0, 2, 1).contiguous().view(batch_size, 256 * self.k)  # (b, n*k)
        p_ = p.permute(0, 2, 1).contiguous().view(batch_size, 256 * self.k)  # (b, n*k)
        outLinear1 = self.classifier(mu_)   # (b, num_classes)
        # outLinear1 = self.classifier(samples_latent)   # (b, num_classes)

        # # classification after attention module
        # mu_ = mu.permute(0, 2, 1).contiguous().view(batch_size, 256 * self.k)  # (b, n*k)
        # sigmap_ = sigmap.permute(0, 2, 1).contiguous().view(batch_size, 256 * self.k)  # (b, n*k)
        # p_ = p.permute(0, 2, 1).contiguous().view(batch_size, 256 * self.k)  # (b, n*k)
        # sigmap_hat = self.sigmap_fc(sigmap_)
        # p_hat = self.p_fc(p_)
        # sigmap_p_hat = sigmap_hat + p_hat
        # sigmap_p_hat = self.sigmap_p_fc(sigmap_p_hat)   # (b, k)
        # sigmap_p_hat = sigmap_p_hat.unsqueeze(1).repeat(1, self.n, 1)   # (b, n, k)
        # mu_hat = mu * sigmap_p_hat   # (b, n, k)
        # mu_hat = mu_hat.permute(0, 2, 1).contiguous().view(batch_size, 256 * self.k)   # (b, n*k)
        # outLinear1 = self.classifier(mu_hat)   # (b, num_classes)

        return outLinear1, rec, mu_, sigmap_, p_   # for training and testing

        # return outLinear1, rec, mu_, sigmap_, p_, x_re   # for generating images

        # return outLinear1_samples, outLinear1_samples_rec, rec, p_loss_batch, mu_loss_batch, sigmap_loss_batch, x_re   # for evaluation in try20210619

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class PROB_LATENT(nn.Module):
    def __init__(self, num_classes, num_channels, im_size, K=4, n=256):
        super().__init__()
        self.k = K
        self.n = n
        self.relu = nn.ReLU(inplace=True)
        self.x_mu_layer = nn.Linear(768, K * n)
        self.x_sigmap_layer = nn.Linear(768, K * n)
        self.x_p_layer = nn.Linear(768, K * n)

    def forward(self, latent_concat):

        x_mu = self.x_mu_layer(latent_concat)
        x_sigmap = self.x_sigmap_layer(latent_concat)
        x_p = self.x_p_layer(latent_concat)

        x_sigmap = F.softplus(x_sigmap) + 1e-8    # >0
        x_sigmap = torch.where(x_sigmap > 4, (4 * torch.ones_like(x_sigmap)).cuda(), x_sigmap)
        x_sigmap = torch.where(x_sigmap < 0.001, (0.001 * torch.ones_like(x_sigmap)).cuda(), x_sigmap)
        x_p = F.softplus(x_p) + 1e-8 + 0.1     # > 0.1
        x_p = torch.where(x_p > 10, (10 * torch.ones_like(x_p)).cuda(), x_p)

        # x_K_choose = torch.randint(0, self.k, (x_mu.shape[0], self.n))
        x_K_choose = torch.randint(0, self.k, (x_mu.shape[0], 1)).repeat(1, self.n)
        # x_K_choose = x_K_choose.unsqueeze(1).repeat(1, n)
        x_K_choose_one_hot = torch.Tensor(x_mu.shape[0] * self.n, self.k)
        x_K_choose_one_hot.zero_()
        x_K_choose_one_hot.scatter_(1, x_K_choose.long().view(-1, 1), 1)  # one-hot encoding
        x_K_choose_one_hot = x_K_choose_one_hot.view(x_mu.shape[0], self.n, self.k).cuda()

        x_mu = x_mu.view(-1, self.n, self.k)
        x_sigmap = x_sigmap.view(-1, self.n, self.k)
        x_p = x_p.view(-1, self.n, self.k)

        # samples_x = ut.newsample_batchall(x_mu, x_sigmap, x_p, x_K_choose_one_hot)
        samples_x = ut.newsample_batchall_try20210622(x_mu, x_sigmap, x_p, x_K_choose_one_hot)

        return samples_x, x_mu, x_sigmap, x_p   # (b, n), (b, n, k)

