import os
import numpy as np
import torch
# from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
from torch.utils.data import Dataset
from torch import Tensor
from typing import Tuple

class CIFAR10(datasets.CIFAR10):
	def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
		super(CIFAR10, self).__init__(root, train=train, transform=transform,
									  target_transform=target_transform, download=download)

		# unify the interface
		if not hasattr(self, 'data'):  # torch <= 0.4.1
			if self.train:
				self.data, self.targets = self.train_data, self.train_labels
			else:
				self.data, self.targets = self.test_data, self.test_labels

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]

		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		img = Image.fromarray(img)

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target, index

	@property
	def num_classes(self):
		return 10


class CIFAR100(datasets.CIFAR100):
	def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
		super(CIFAR100, self).__init__(root, train=train, transform=transform,
									   target_transform=target_transform, download=download)

		# unify the interface
		if not hasattr(self, 'data'):  # torch <= 0.4.1
			if self.train:
				self.data, self.targets = self.train_data, self.train_labels
			else:
				self.data, self.targets = self.test_data, self.test_labels

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]

		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		img = Image.fromarray(img)

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target, index

	@property
	def num_classes(self):
		return 100


class TensorDataset(Dataset[Tuple[Tensor, ...]]):
	r"""Dataset wrapping tensors.

	Each sample will be retrieved by indexing tensors along the first dimension.

	Args:
		*tensors (Tensor): tensors that have the same size of the first dimension.
	"""
	tensors: Tuple[Tensor, ...]

	def __init__(self, *tensors: Tensor) -> None:
		assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
		self.tensors = tensors

	def __getitem__(self, index):
		return tuple(tensor[index] for tensor in self.tensors), index

	def __len__(self):
		return self.tensors[0].size(0)