# evaluation of macro-F1 scores of MoEP-AE-OSR
import torch
import torch.nn as nn
import json
import torchvision.transforms as tf
import openSetClassifier_MoEP_AE
from utils_MoEP_AE import gather_outputs_OOD
import torchvision
import numpy as np
import sklearn
import sklearn.metrics


# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create dataloaders for training
print('==> Preparing data..')
with open('config_data.json') as config_file:
	cfg = json.load(config_file)['CIFAR10']


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


def softmax(x):
	if len(x.shape) > 1:
		tmp = np.max(x, axis=1)
		x -= tmp.reshape((x.shape[0], 1))
		x = np.exp(x)
		tmp = np.sum(x, axis=1)
		x /= tmp.reshape((x.shape[0], 1))
	else:
		tmp = np.max(x)
		x -= tmp
		x = np.exp(x)
		tmp = np.sum(x)
		x /= tmp
	return x


trainloader, valloader, testloader, trainSet = get_train_loaders(cfg)
# CIFAR10_mean, CIFAR10_std = get_mean_std(trainSet)
testout_transform = tf.Compose([
		tf.Resize(cfg['im_size']),
		tf.ToTensor(),
		# tf.Normalize((CIFAR10_mean, CIFAR10_std)),
	])
testout_dataset1 = torchvision.datasets.ImageFolder("datasets/OOD_datasets/Imagenet_crop", transform=testout_transform)
testout_dataloader1 = torch.utils.data.DataLoader(testout_dataset1, batch_size=50, shuffle=False, num_workers=2)
testout_dataset2 = torchvision.datasets.ImageFolder("datasets/OOD_datasets/Imagenet_resize", transform=testout_transform)
testout_dataloader2 = torch.utils.data.DataLoader(testout_dataset2, batch_size=50, shuffle=False, num_workers=2)
testout_dataset3 = torchvision.datasets.ImageFolder("datasets/OOD_datasets/LSUN_crop", transform=testout_transform)
testout_dataloader3 = torch.utils.data.DataLoader(testout_dataset3, batch_size=50, shuffle=False, num_workers=2)
testout_dataset4 = torchvision.datasets.ImageFolder("datasets/OOD_datasets/LSUN_resize", transform=testout_transform)
testout_dataloader4 = torch.utils.data.DataLoader(testout_dataset4, batch_size=50, shuffle=False, num_workers=2)

print('==> Building network..')
net = openSetClassifier_MoEP_AE.openSetClassifier(10, cfg['im_channels'], cfg['im_size'], init_weights=1, dropout=cfg['dropout'])

net = net.to(device)

# Load checkpoint.
print('==> Resuming from checkpoint..')
checkpoint = torch.load('weights/CIFAR10/CIFAR10_MoEP-AE_macro_F1_scores.pth')
net_dict = net.state_dict()
pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in net_dict}
net_dict.update(pretrained_dict)
net.load_state_dict(net_dict)

net.eval()


with torch.no_grad():

	xval, _ = gather_outputs_OOD(net, valloader, data_idx=0, calculate_scores=False, num_classes=10)   # (b*K, num_classes)
	xin, yin = gather_outputs_OOD(net, testloader, data_idx=0, calculate_scores=False, num_classes=10)   # (b*K, num_classes)
	xout1, _ = gather_outputs_OOD(net, testout_dataloader1, data_idx=0, calculate_scores=False, unknown=True, num_classes=10)
	xout2, _ = gather_outputs_OOD(net, testout_dataloader2, data_idx=0, calculate_scores=False, unknown=True, num_classes=10)
	xout3, _ = gather_outputs_OOD(net, testout_dataloader3, data_idx=0, calculate_scores=False, unknown=True, num_classes=10)
	xout4, _ = gather_outputs_OOD(net, testout_dataloader4, data_idx=0, calculate_scores=False, unknown=True, num_classes=10)

	# ################################ calculate the auroc ###############################################
	# # auroc = metrics.auroc2(xK_final, xU_final)
	# auroc = metrics.auroc2(xK, xU)
	# print('the auroc now is : ', auroc)
	# print('the best auroc is : ', best_auroc)
	# accuracy = metrics.accuracy2(xK, yK)
	# print('the accuracy now is : ', accuracy)

	# ################################# calculate the macro-F1 score using the logits ########################################
	# calculate the threshold
	val_scores = np.max(xval, 1)
	val_scores.sort()
	threshold = val_scores[int(0.1 * len(val_scores)) - 1]  # to make sure the 90% validation datasets are judged as the known samples
	# calculate the test results of in-distribution data and out-of-distribution data
	test_results_part1 = np.argmax(xin, 1)
	test_results_part1[np.where(np.max(xin, 1) <= threshold)] = 10
	test_results_part2_out1 = 10 * np.ones(xout1.shape[0])
	test_results_part2_out1[np.where(np.max(xout1, 1) > threshold)] = np.argmax(xout1, 1)[np.where(np.max(xout1, 1) > threshold)]
	test_results_out1 = np.concatenate((test_results_part1, test_results_part2_out1))
	test_results_part2_out2 = 10 * np.ones(xout2.shape[0])
	test_results_part2_out2[np.where(np.max(xout2, 1) > threshold)] = np.argmax(xout2, 1)[np.where(np.max(xout2, 1) > threshold)]
	test_results_out2 = np.concatenate((test_results_part1, test_results_part2_out2))
	test_results_part2_out3 = 10 * np.ones(xout3.shape[0])
	test_results_part2_out3[np.where(np.max(xout3, 1) > threshold)] = np.argmax(xout3, 1)[np.where(np.max(xout3, 1) > threshold)]
	test_results_out3 = np.concatenate((test_results_part1, test_results_part2_out3))
	test_results_part2_out4 = 10 * np.ones(xout4.shape[0])
	test_results_part2_out4[np.where(np.max(xout4, 1) > threshold)] = np.argmax(xout4, 1)[np.where(np.max(xout4, 1) > threshold)]
	test_results_out4 = np.concatenate((test_results_part1, test_results_part2_out4))
	# calculate the macro-F1 score by sklearn
	true_results1 = np.concatenate((yin, 10 * np.ones(xout1.shape[0])))
	true_results2 = np.concatenate((yin, 10 * np.ones(xout2.shape[0])))
	true_results3 = np.concatenate((yin, 10 * np.ones(xout3.shape[0])))
	true_results4 = np.concatenate((yin, 10 * np.ones(xout4.shape[0])))
	macro_F1_out1 = sklearn.metrics.f1_score(true_results1, test_results_out1, average="macro")
	macro_F1_out2 = sklearn.metrics.f1_score(true_results2, test_results_out2, average="macro")
	macro_F1_out3 = sklearn.metrics.f1_score(true_results3, test_results_out3, average="macro")
	macro_F1_out4 = sklearn.metrics.f1_score(true_results4, test_results_out4, average="macro")
	print('the test macro-F1 score of Imagenet_crop is : ', macro_F1_out1)
	print('the test macro-F1 score of Imagenet_resize is : ', macro_F1_out2)
	print('the test macro-F1 score of LSUN_crop is : ', macro_F1_out3)
	print('the test macro-F1 score of LSUN_resize is : ', macro_F1_out4)

	# # ###################### calculate the macro-F1 score using the probabilities after softmax ########################
	# # calculate the threshold
	# val_scores = np.max(softmax(xval), 1)
	# val_scores.sort()
	# threshold = val_scores[int(0.1 * len(val_scores)) - 1]   # to make sure the 90% validation datasets are judged as the known samples
	# # calculate the test results of in-distribution data and out-of-distribution data
	# test_results_part1 = np.argmax(softmax(xin), 1)
	# test_results_part1[np.where(np.max(softmax(xin), 1) <= threshold)] = 10
	# test_results_part2_out1 = 10 * np.ones(xout1.shape[0])
	# test_results_part2_out1[np.where(np.max(softmax(xout1), 1) > threshold)] = np.argmax(softmax(xout1), 1)[np.where(np.max(softmax(xout1), 1) > threshold)]
	# test_results_out1 = np.concatenate((test_results_part1, test_results_part2_out1))
	# test_results_part2_out2 = 10 * np.ones(xout2.shape[0])
	# test_results_part2_out2[np.where(np.max(softmax(xout2), 1) > threshold)] = np.argmax(softmax(xout2), 1)[np.where(np.max(softmax(xout2), 1) > threshold)]
	# test_results_out2 = np.concatenate((test_results_part1, test_results_part2_out2))
	# test_results_part2_out3 = 10 * np.ones(xout3.shape[0])
	# test_results_part2_out3[np.where(np.max(softmax(xout3), 1) > threshold)] = np.argmax(softmax(xout3), 1)[np.where(np.max(softmax(xout3), 1) > threshold)]
	# test_results_out3 = np.concatenate((test_results_part1, test_results_part2_out3))
	# test_results_part2_out4 = 10 * np.ones(xout4.shape[0])
	# test_results_part2_out4[np.where(np.max(softmax(xout4), 1) > threshold)] = np.argmax(softmax(xout4), 1)[np.where(np.max(softmax(xout4), 1) > threshold)]
	# test_results_out4 = np.concatenate((test_results_part1, test_results_part2_out4))
	# # calculate the macro-F1 score by sklearn
	# true_results1 = np.concatenate((yin, 10 * np.ones(xout1.shape[0])))
	# true_results2 = np.concatenate((yin, 10 * np.ones(xout2.shape[0])))
	# true_results3 = np.concatenate((yin, 10 * np.ones(xout3.shape[0])))
	# true_results4 = np.concatenate((yin, 10 * np.ones(xout4.shape[0])))
	# macro_F1_out1 = sklearn.metrics.f1_score(true_results1, test_results_out1, average="macro")
	# macro_F1_out2 = sklearn.metrics.f1_score(true_results2, test_results_out2, average="macro")
	# macro_F1_out3 = sklearn.metrics.f1_score(true_results3, test_results_out3, average="macro")
	# macro_F1_out4 = sklearn.metrics.f1_score(true_results4, test_results_out4, average="macro")
	# print('the test macro-F1 score of Imagenet_crop is : ', macro_F1_out1)
	# print('the test macro-F1 score of Imagenet_resize is : ', macro_F1_out2)
	# print('the test macro-F1 score of LSUN_crop is : ', macro_F1_out3)
	# print('the test macro-F1 score of LSUN_resize is : ', macro_F1_out4)


