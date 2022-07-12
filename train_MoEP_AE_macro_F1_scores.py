# coding:utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import json
import torchvision.transforms as tf
import argparse
import openSetClassifier_MoEP_AE
from utils_MoEP_AE import progress_bar
import os
# from utils_MoEP_AE_ResNet18_CEloss import gather_outputs
from utils_MoEP_AE import gather_outputs_OOD
import metrics
import torchvision
from torchvision import models
import numpy as np
import sklearn


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description='Open Set Classifier Training')
# parser.add_argument('--dataset', required=True, type=str, help='Dataset for training',
# 									choices=['MNIST', 'SVHN', 'CIFAR10', 'CIFAR+10', 'CIFAR+50', 'TinyImageNet'])
# parser.add_argument('--trial', required=True, type=int, help='Trial number, 0-4 provided')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from the checkpoint')
parser.add_argument('--alpha', default=10, type=int, help='Magnitude of the anchor point')
parser.add_argument('--lbda', default=0.1, type=float, help='Weighting of Anchor loss component')
parser.add_argument('--tensorboard', '-t', action='store_true', help='Plot on tensorboardX')
# parser.add_argument('--name', default="MoEP-AE_ResNet18_CEloss_unified_try20210707_swin_vit_para_CIFAR10", type=str, help='Optional name for saving and tensorboard')
parser.add_argument('--name', default="MoEP-AE_newtry_CIFAR10", type=str, help='Optional name for saving and tensorboard')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters useful when resuming and finetuning
best_acc = 0
best_loss = 10000
best_auroc = 0
best_F1 = 0
start_epoch = 0

# Create dataloaders for training
print('==> Preparing data..')
# with open('datasets/config_ResNet18_20210622.json') as config_file:
# with open('datasets/config_ResNet18_20210707_swin_vit.json') as config_file:
with open('datasets/config_data.json') as config_file:
# with open('datasets/config_ResNet18_32_32.json') as config_file:
	cfg = json.load(config_file)['CIFAR10']


class Net1(nn.Module):
    def __init__(self, model):  # 此处的model参数是已经加载了预训练参数的模型，方便继承预训练成果
        super(Net1, self).__init__()
        self.extracted_layers = nn.Sequential(*list(model.children())[:-4])
    def forward(self, x):
        x = self.extracted_layers(x)
        return x


class Net2(nn.Module):
    def __init__(self, model):  # 此处的model参数是已经加载了预训练参数的模型，方便继承预训练成果
        super(Net2, self).__init__()
        self.extracted_layers = nn.Sequential(*list(model.children())[-4:])
    def forward(self, x):
        x = self.extracted_layers(x)
        return x


class Net(nn.Module):
    def __init__(self, model, num_class):  # 此处的model参数是已经加载了预训练参数的模型，方便继承预训练成果
        super(Net, self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.ln1 = nn.Linear(512, 4096)
        self.ln2 = nn.Linear(4096, 4096)
        self.ln3 = nn.Linear(4096, num_class)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.resnet_layer(x)
        x1 = x.view(x.shape[0], -1)
        x2 = self.relu(self.ln1(x1))
        x2 = self.relu(self.ln2(x2))
        x2 = self.ln3(x2)
        return x1, x2   # feature and output


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def load_datasets(cfg):

	# controls data transforms
	flip = cfg['data_transforms']['flip']
	rotate = cfg['data_transforms']['rotate']
	scale_min = cfg['data_transforms']['scale_min']
	means_imagenet = [0.5, 0.5, 0.5]
	stds_imagenet = [0.5, 0.5, 0.5]

	transforms = {
	    'train': tf.Compose([
	        tf.Resize(cfg['im_size']),
	        tf.RandomResizedCrop(cfg['im_size'], scale = (scale_min, 1.0)),
	        tf.RandomHorizontalFlip(flip),
	        tf.RandomRotation(rotate),
	        tf.ToTensor(),
	        # tf.Normalize(means, stds)
	        # tf.Normalize(means_imagenet, stds_imagenet)
	    ]),
	    'val': tf.Compose([
	        tf.Resize(cfg['im_size']),
	        tf.ToTensor(),
	        # tf.Normalize(means, stds)
	        # tf.Normalize(means_imagenet, stds_imagenet)
	    ]),
	    'test': tf.Compose([
	        tf.Resize(cfg['im_size']),
	        tf.ToTensor(),
	        # tf.Normalize(means, stds)
	        # tf.Normalize(means_imagenet, stds_imagenet)
	    ])
	}

	trainSet = torchvision.datasets.CIFAR10('datasets/data', transform = transforms['train'], download = True)
	valSet = torchvision.datasets.CIFAR10('datasets/data', transform = transforms['val'])
	testSet = torchvision.datasets.CIFAR10('datasets/data', train = False, transform = transforms['test'])

	return trainSet, valSet, testSet


def create_dataSubsets(dataset, idxs_to_use=None):

	# ##################################### for pytorch 1.X ##############################################
	# get class label for dataset images. svhn has different syntax as .labels
	targets = dataset.targets
	# ####################################################################################################
	subset_idxs = []
	if idxs_to_use == None:
		for i, lbl in enumerate(targets):
			subset_idxs += [i]
	else:
		for class_num in idxs_to_use.keys():
			subset_idxs += idxs_to_use[class_num]

	dataSubset = torch.utils.data.Subset(dataset, subset_idxs)
	return dataSubset


def get_train_loaders(cfg):

	trainSet, valSet, testSet = load_datasets(cfg)

	with open("datasets/{}/trainval_idxs.json".format('CIFAR10')) as f:
		trainValIdxs = json.load(f)
		train_idxs = trainValIdxs['Train']
		val_idxs = trainValIdxs['Val']

	trainSubset = create_dataSubsets(trainSet, train_idxs)
	valSubset = create_dataSubsets(valSet, val_idxs)
	testSubset = create_dataSubsets(testSet)

	batch_size = cfg['batch_size']

	trainloader = torch.utils.data.DataLoader(trainSubset, batch_size=batch_size, shuffle=True, num_workers=cfg['dataloader_workers'])
	valloader = torch.utils.data.DataLoader(valSubset, batch_size=batch_size, shuffle=True)
	testloader = torch.utils.data.DataLoader(testSubset, batch_size=batch_size, shuffle=True)

	return trainloader, valloader, testloader, trainSet


def get_mean_std(dataset, ratio=0.01):
	"""
	Get mean and std by sample ratio
	"""
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset)*ratio), shuffle=True, num_workers=2)
	train = iter(dataloader).next()[0]   # the data on one batch
	mean = np.mean(train.numpy(), axis=(0, 2, 3))
	std = np.std(train.numpy(), axis=(0, 2, 3))
	return mean, std


trainloader, valloader, testloader, trainSet = get_train_loaders(cfg)
# CIFAR10_mean, CIFAR10_std = get_mean_std(trainSet)
testout_transform = tf.Compose([
		tf.Resize(cfg['im_size']),
		tf.ToTensor(),
		# tf.Normalize((CIFAR10_mean, CIFAR10_std)),
	])
testout_dataset = torchvision.datasets.ImageFolder("datasets/OOD_datasets/Imagenet_crop", transform=testout_transform)
testout_dataloader = torch.utils.data.DataLoader(testout_dataset, batch_size=50, shuffle=False, num_workers=2)

print('==> Building network..')
print(not args.resume)


net = openSetClassifier_MoEP_AE.openSetClassifier(10, cfg['im_channels'], cfg['im_size'], init_weights=not args.resume, dropout=cfg['dropout'])
net.apply(inplace_relu)
net.swin_vit.load_state_dict(torch.load('swin_base_patch4_window7_224_22k.pth')['model'], strict=False)   # use the pretrained model trained on ImageNet


# # load part of the pre-trained parameters
# model_dict1 = net.state_dict()
# # try:
#     # model_2_ = torch.load('../improved-wgan-pytorch-master/networks/weights/{}/{}_{}_try3_F_and_C_unifiedAccuracy.pth'.format(args.dataset, args.dataset, args.trial))
# model_2 = torch.load('networks/weights/{}/{}_{}_MoEP-AE_ResNet18_CEloss_unified_try20210623_vit_onlyencoderCEclassifierAUROC.pth'.format(args.dataset, args.dataset, args.trial))
# # except:
# #     # model_2_ = torch.load('../../improved-wgan-pytorch-master/networks/weights/{}/{}_{}_try3_F_and_C_unifiedAccuracy.pth'.format(args.dataset, args.dataset, args.trial))
# #     model_2_ = torch.load('networks/weights/{}/{}_{}_try3_F_and_C_unifiedAccuracy.pth'.format(args.dataset, args.dataset, args.trial))
# # model_2 = Net1(model_2_)
# # model_dict2 = model_2.state_dict()
# model_dict2 = model_2['net']
# model_list1 = list(model_dict1.keys())
# model_list2 = list(model_dict2.keys())
# len1 = len(model_list1)
# len2 = len(model_list2)
# minlen = min(len1, len2)
# for n in range(minlen):
#     # print(model_dict1[model_list1[n]])
#     if model_dict1[model_list1[n]].shape != model_dict2[model_list2[n]].shape:
#         continue
#     model_dict1[model_list1[n]] = model_dict2[model_list2[n]]
# net.load_state_dict(model_dict1)


net = net.to(device)
training_iter = int(args.resume)

net.train()

# # optimizer = optim.SGD(net.parameters(), lr=cfg['openset_training']['learning_rate'][training_iter],
# optimizer = optim.SGD(net.parameters(), lr=0.01,   # no finetuning
# # optimizer = optim.SGD(net.parameters(), lr=0.001,   # finetuning1
# # optimizer = optim.SGD(net.parameters(), lr=0.0001,   # finetuning2
# 							momentum=0.9, weight_decay=cfg['openset_training']['weight_decay'])
# # optimizer = optim.SGD(net.parameters(), lr=cfg['openset_training']['learning_rate'][training_iter],
# # 							momentum=0.9)
# # optimizer = optim.Adam(net.parameters(), lr=cfg['openset_training']['learning_rate'][training_iter], weight_decay=cfg['openset_training']['weight_decay'])

# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=cfg['openset_training']['weight_decay'])
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=cfg['openset_training']['weight_decay'])
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=cfg['openset_training']['weight_decay'])
# optimizer = optim.AdamW(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
optimizer = optim.AdamW(net.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
# optimizer = optim.AdamW(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
# optimizer = optim.AdamW(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
# optimizer = optim.SGD(net.parameters(), lr=0.001)
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=cfg['openset_training']['weight_decay'])
# optimizer = optim.Adam(net.parameters(), lr=0.001)
# optimizer = optim.Adam(net.parameters(), lr=0.0001)


# Training
def train(epoch):

	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	cac_loss = 0
	rec_loss = 0
	kl_loss = 0
	correctDist = 0
	total = 0

	# if epoch < 150:   # for CIFAR/CIFAR+
	# # # if epoch < 500:   # for TinyImageNet
	# 	optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=cfg['openset_training']['weight_decay'])
	# elif epoch < 300:   # for CIFAR/CIFAR+
	# # # elif epoch < 800:   # for TinyImageNet
	# 	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=cfg['openset_training']['weight_decay'])
	# else:
	# 	optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=cfg['openset_training']['weight_decay'])

	for batch_idx, (inputs, targets) in enumerate(trainloader):
		print(batch_idx)
		# if batch_idx == 1:
		# 	sys.exit(0)

		inputs = inputs.to(device)

		if inputs.shape[1] != 3:
			inputs = inputs.repeat(1, 3, 1, 1)

		# convert from original dataset label to known class label
		# targets = torch.Tensor([mapping[x] for x in targets]).long()
		targets = targets.long()
		target_en = torch.Tensor(targets.shape[0], 10)
		target_en.zero_()
		target_en.scatter_(1, targets.view(-1, 1), 1)  # one-hot encoding
		target_en = target_en.to(device)
		targets = targets.to(device)

		optimizer.zero_grad()

		outLinear1, rec, mu, sigmap, p = net(inputs, target_en)   # outLinear1 size:(b, num_classes)

		# # # ########################## calculate the final classification results #########################
		# # # method1: use the mean of the K results
		# # outLinear1_mu_final = 0
		# # for i in range(k):
		# # 	outLinear1_mu_final += outLinear1_mu[i::k, :]
		# # outLinear1_mu_final = outLinear1_mu_final / k
		# # # ###############################################################################################
		# #
		# # # calculate the CEloss
		# # ceLoss_mu = 0
		# # for i in range(k):
		# # 	outLinear1_mu_ = outLinear1_mu[i::k, :]
		# # 	ceLoss_mu += nn.functional.cross_entropy(outLinear1_mu_, targets)
		# # ceLoss_mu = ceLoss_mu / k
		#
		# # ########################## calculate the final classification results #########################
		# # method1: use the mean of the K results
		# outLinear1_samples_final = 0
		# for i in range(k):
		# 	outLinear1_samples_final += outLinear1_samples[i::k, :]
		# outLinear1_samples_final = outLinear1_samples_final / k
		# # ###############################################################################################
		#
		# # calculate the CEloss
		# ceLoss_samples = 0
		# for i in range(k):
		# 	outLinear1_samples_ = outLinear1_samples[i::k, :]
		# 	ceLoss_samples += nn.functional.cross_entropy(outLinear1_samples_, targets)
		# ceLoss_samples = ceLoss_samples / k

		# ###################### calculate the distance matrix in a batch and the two masks ############################
		# calculate the distance matrice
		mu = mu.float()
		mu_dist = torch.norm(mu[:, None] - mu, dim=2, p=2)   # (b, b)
		sigmap = sigmap.float()
		sigmap_dist = torch.norm(sigmap[:, None] - sigmap, dim=2, p=2)   # (b, b)
		p = p.float()
		p_dist = torch.norm(p[:, None] - p, dim=2, p=2)   # (b, b)

		# calculate the masks
		targets_ = targets.unsqueeze(1).float()
		mask_same = (torch.norm(targets_[:, None] - targets_, dim=2, p=2)==0).float()   # (b, b)
		mask_diff = 1 - mask_same   # (b, b)

		# calculate the loss of this part
		mu_loss = torch.mean(mu_dist * mask_same) / torch.mean(mu_dist * mask_diff)
		sigmap_loss = torch.mean(sigmap_dist * mask_same) / torch.mean(sigmap_dist * mask_diff)
		p_loss = torch.mean(p_dist * mask_same) / torch.mean(p_dist * mask_diff)
		# mu_loss = torch.mean(mu_dist * mask_same) - torch.mean(mu_dist * mask_diff)
		# sigmap_loss = torch.mean(sigmap_dist * mask_same) - torch.mean(sigmap_dist * mask_diff)
		# p_loss = torch.mean(p_dist * mask_same) - torch.mean(p_dist * mask_diff)

		# ##############################################################################################################

		ceLoss = nn.functional.cross_entropy(outLinear1, targets)

		total_loss = 1 * ceLoss.double() + 1 * rec.double() + 0.1 * mu_loss.double() + 0.1 * sigmap_loss.double() + 0.1 * p_loss.double()

		# for ablation study
		# total_loss = 1 * ceLoss.double()
		# total_loss = 1 * ceLoss.double() + 1 * rec.double()

		total_loss.backward()

		# nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
		nn.utils.clip_grad_norm_(net.parameters(), max_norm=2000, norm_type=2)
		# nn.utils.clip_grad_value_(net.parameters(), 20)  # 直接做梯度裁剪

		optimizer.step()

		train_loss = train_loss + total_loss.detach().cpu().numpy()

		_, predicted = outLinear1.max(1)

		# total += targets.size(0)
		total = total + targets.size(0)
		# correctDist += predicted.eq(targets).sum().item()
		correctDist = correctDist + predicted.eq(targets).sum().item()

		# progress_bar(batch_idx, len(trainloader), 'Total_Loss: %.3f|CE_Loss: %.3f|rec_Loss: %.3f|Acc: %.3f%% (%d/%d)'
		# 	% (total_loss, ceLoss, rec, 100.*correctDist/total, correctDist, total))
		progress_bar(batch_idx, len(trainloader), 'Total_Loss: %.3f|CE_Loss: %.3f|rec_Loss: %.3f|Mu_Loss: %.3f|Sigmap_Loss: %.3f|P_Loss: %.3f|Acc: %.3f%% (%d/%d)'
			% (total_loss, ceLoss, rec, mu_loss, sigmap_loss, p_loss, 100.*correctDist/total, correctDist, total))


def val(epoch):
	global best_loss
	global best_acc
	global best_auroc
	global best_F1
	net.eval()
	anchor_loss = 0
	cac_loss = 0
	rec_loss = 0
	kl_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(valloader):
			inputs = inputs.to(device)

			if inputs.shape[1] != 3:
				inputs = inputs.repeat(1, 3, 1, 1)

			# targets = torch.Tensor([mapping[x] for x in targets]).long()
			targets = targets.long()

			target_en = torch.Tensor(targets.shape[0], 10)
			target_en.zero_()
			target_en.scatter_(1, targets.view(-1, 1), 1)  # one-hot encoding
			target_en = target_en.to(device)
			targets = targets.to(device)

			outLinear1, rec, mu, sigmap, p = net(inputs, target_en)

			# k = int(outLinear1_samples.shape[0] / target_en.shape[0])
			# k = int(outLinear1_mu.shape[0] / target_en.shape[0])

			# # ########################## calculate the final classification results #########################
			# # method1: use the mean of the K results
			# outLinear1_samples_final = 0
			# for i in range(k):
			# 	outLinear1_samples_final += outLinear1_samples[i::k, :]
			# outLinear1_samples_final = outLinear1_samples_final / k
			# # ###############################################################################################
			#
			# # # calculate the CEloss
			# # ceLoss_mu = 0
			# # for i in range(k):
			# # 	outLinear1_mu_ = outLinear1_mu[i::k, :]
			# # 	ceLoss_mu += nn.functional.cross_entropy(outLinear1_mu_, targets)
			# # ceLoss_mu = ceLoss_mu / k
			#
			# # calculate the CEloss
			# ceLoss_samples = 0
			# for i in range(k):
			# 	outLinear1_samples_ = outLinear1_samples[i::k, :]
			# 	ceLoss_samples += nn.functional.cross_entropy(outLinear1_samples_, targets)
			# ceLoss_samples = ceLoss_samples / k

			# ###################### calculate the distance matrix in a batch and the two masks ############################
			# calculate the distance matrice
			mu = mu.float()
			mu_dist = torch.norm(mu[:, None] - mu, dim=2, p=2)  # (b, b)
			sigmap = sigmap.float()
			sigmap_dist = torch.norm(sigmap[:, None] - sigmap, dim=2, p=2)  # (b, b)
			p = p.float()
			p_dist = torch.norm(p[:, None] - p, dim=2, p=2)  # (b, b)

			# calculate the masks
			targets_ = targets.unsqueeze(1).float()
			mask_same = (torch.norm(targets_[:, None] - targets_, dim=2, p=2) == 0).float()  # (b, b)
			mask_diff = 1 - mask_same  # (b, b)

			# calculate the loss of this part
			mu_loss = torch.mean(mu_dist * mask_same) / torch.mean(mu_dist * mask_diff)
			sigmap_loss = torch.mean(sigmap_dist * mask_same) / torch.mean(sigmap_dist * mask_diff)
			p_loss = torch.mean(p_dist * mask_same) / torch.mean(p_dist * mask_diff)
			# mu_loss = torch.mean(mu_dist * mask_same) - torch.mean(mu_dist * mask_diff)
			# sigmap_loss = torch.mean(sigmap_dist * mask_same) - torch.mean(sigmap_dist * mask_diff)
			# p_loss = torch.mean(p_dist * mask_same) - torch.mean(p_dist * mask_diff)

			# ##############################################################################################################

			ceLoss = nn.functional.cross_entropy(outLinear1, targets)

			# total_loss = 1 * ceLoss.double() + 100 * rec.double()
			# total_loss = 1 * ceLoss.double() + 100 * rec.double() + 1 * mu_loss.double() + 1 * sigmap_loss.double() + 1 * p_loss.double()
			# total_loss = 100 * ceLoss.double() + 1 * rec.double() + 1 * mu_loss.double() + 1 * sigmap_loss.double() + 1 * p_loss.double()
			total_loss = 1 * ceLoss.double() + 1 * rec.double() + 0.1 * mu_loss.double() + 0.1 * sigmap_loss.double() + 0.1 * p_loss.double()
			# total_loss = 1 * ceLoss + 100 * rec + 1 * mu_loss + 1 * sigmap_loss + 1 * p_loss   # for CIFAR10, CIFAR+10
			# total_loss = 100 * ceLoss + 1 * rec + 0.1 * mu_loss + 0.1 * sigmap_loss + 0.1 * p_loss   # for CIFAR10, CIFAR+10
			# total_loss = 1 * ceLoss + 1 * rec + 1 * mu_loss + 1 * sigmap_loss + 1 * p_loss   # for SVHN, MNIST
			# total_loss = 1 * rec.double()
			total_loss = total_loss.detach().cpu().numpy()

			_, predicted = outLinear1.max(1)

			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			progress_bar(batch_idx, len(valloader), 'Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

		total_loss /= len(valloader)
		print(total_loss)
		acc = 100.*correct/total

		# calculate the auroc
		# anchor_means = find_anchor_means(net, mapping, args.dataset, args.trial, cfg, only_correct=True)
		# net.set_anchors(torch.Tensor(anchor_means))
		# xK, yK = gather_outputs(net, mapping, knownloader, data_idx=1, calculate_scores=True, num_classes=cfg['num_known_classes'])
		# xU, yU = gather_outputs(net, mapping, unknownloader, data_idx=1, calculate_scores=True, unknown=True, num_classes=cfg['num_known_classes'])
		# xK, yK = gather_outputs(net, mapping, knownloader, data_idx=0, calculate_scores=False, num_classes=cfg['num_known_classes'])   # (b*K, num_classes)
		# xU, _ = gather_outputs(net, mapping, unknownloader, data_idx=0, calculate_scores=False, unknown=True, num_classes=cfg['num_known_classes'])
		xval, _ = gather_outputs_OOD(net, valloader, data_idx=0, calculate_scores=False, num_classes=10)   # (b*K, num_classes)
		xin, yin = gather_outputs_OOD(net, testloader, data_idx=0, calculate_scores=False, num_classes=10)   # (b*K, num_classes)
		xout, _ = gather_outputs_OOD(net, testout_dataloader, data_idx=0, calculate_scores=False, unknown=True, num_classes=10)

		# ################################ calculate the auroc ###############################################
		# # auroc = metrics.auroc2(xK_final, xU_final)
		# auroc = metrics.auroc2(xK, xU)
		# print('the auroc now is : ', auroc)
		# print('the best auroc is : ', best_auroc)
		# accuracy = metrics.accuracy2(xK, yK)
		# print('the accuracy now is : ', accuracy)

		# #################################calculate the macro-F1 score ########################################
		# calculate the threshold
		val_scores = np.max(xval, 1)
		val_scores.sort()
		threshold = val_scores[int(0.1 * len(val_scores)) - 1]  # to make sure the 90% validation datasets are judged as the known samples
		# calculate the test results of in-distribution data and out-of-distribution data
		test_results_part1 = np.argmax(xin, 1)
		test_results_part1[np.where(np.max(xin, 1) <= threshold)] = 10
		test_results_part2 = 10 * np.ones(xout.shape[0])
		test_results_part2[np.where(np.max(xout, 1) > threshold)] = np.argmax(xout, 1)[np.where(np.max(xout, 1) > threshold)]
		test_results = np.concatenate((test_results_part1, test_results_part2))
		# calculate the macro-F1 score by sklearn
		true_results = np.concatenate((yin, 10 * np.ones(xout.shape[0])))
		macro_F1 = sklearn.metrics.f1_score(true_results, test_results, average="macro")
		print('the test macro-F1 score is : ', macro_F1)
		print('the best test macro-F1 score is : ', best_F1)

		# Save checkpoint.
		state = {
			'net': net.state_dict(),
			'acc': acc,
			'epoch': epoch,
		}
		if not os.path.isdir('weights/{}'.format('CIFAR10')):
			os.mkdir('weights/{}'.format('CIFAR10'))

		save_name = '{}_{}CEclassifier'.format('CIFAR10', args.name)

		torch.save(state, 'networks/weights/{}/'.format('CIFAR10')+save_name+'F1.pth')

# max_epoch = 400 + start_epoch
max_epoch = 2000 + start_epoch


for epoch in range(start_epoch, max_epoch):
	train(epoch)
	val(epoch)

