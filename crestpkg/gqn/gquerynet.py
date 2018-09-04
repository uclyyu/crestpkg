import torch, imageio, os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint
from torch.distributions import Normal
from tqdm import tqdm


class ResidualConnect(nn.Module):
	"""Helper class for ResNet."""
	def __init__(self, main, skip):
		super().__init__()
		self.main = main
		self.skip = skip
	def forward(self, inp):
		return self.main(inp) + self.skip(inp)


class PoolSceneEncoder(nn.Module):
	def __init__(self, output_channels, hidden_channels):
		super(PoolSceneEncoder, self).__init__()
		self.output_channels = output_channels
		self.hidden_channesl = hidden_channels
		self.layers = nn.ModuleList([
			nn.Sequential(
				nn.Conv2d(3, output_channels, (2, 2), (2, 2)), 
				nn.ReLU(),
				ResidualConnect(
					nn.Conv2d(output_channels, hidden_channels, (3, 3), (1, 1), padding=1),
					nn.Conv2d(output_channels, hidden_channels, (1, 1), (1, 1), bias=False)),
				nn.ReLU(),
				nn.Conv2d(hidden_channels, output_channels, (2, 2), (2, 2)),
				nn.ReLU()
			),
			nn.Sequential(
				ResidualConnect(
					nn.Conv2d(output_channels+7, hidden_channels, (3, 3), (1, 1), padding=1),
					nn.Conv2d(output_channels+7, hidden_channels, (1, 1), (1, 1), bias=False)), 
				nn.ReLU(),
				nn.Conv2d(hidden_channels, output_channels, (3, 3), (1, 1), padding=1), 
				nn.ReLU(),
				nn.Conv2d(output_channels, output_channels, (1, 1), (1, 1)),
				nn.ReLU(),
				nn.MaxPool2d((16, 16))
			)
		])
		
	def forward(self, views, viewpoints):
		r = self.layers[1](torch.cat([self.layers[0](views), viewpoints.expand(-1, -1, 16, 16)], dim=1))
		return r


class PriorFactor(nn.Module):
	def __init__(self, input_channels, output_channels, hidden_channels=[128, 256]):
		super(PriorFactor, self).__init__()
		self.input_channels = input_channels
		self.output_channels = output_channels 
		self.hidden_channels = hidden_channels
		self.layer = nn.Sequential(
			nn.Conv2d(input_channels, hidden_channels[0], (5, 5), (1, 1), padding=2),
			nn.ReLU(),
			ResidualConnect(
				nn.Conv2d(hidden_channels[0], hidden_channels[1], (5, 5), (1, 1), padding=2),
				nn.Conv2d(hidden_channels[0], hidden_channels[1], (1, 1), (1, 1), bias=False)),
			nn.ReLU(),
			ResidualConnect(
				nn.Conv2d(hidden_channels[1], output_channels, (5, 5), (1, 1), padding=2),
				nn.Conv2d(hidden_channels[1], output_channels, (1, 1), (1, 1), bias=False)),
			nn.ReLU(),
			nn.Conv2d(output_channels, 2 * output_channels, (1, 1), (1, 1)),  # mean, log-variance
		)
		
	def forward(self, inp):
		loc, lva = torch.chunk(self.layer(inp), 2, dim=1)
		scale = (0.5 * lva).exp()
		pd = Normal(loc, scale)
		return pd, loc, lva


class InferenceCoreInlet(nn.Module):
	def __init__(self, output_channels, hidden_channels=[128, 64]):
		super().__init__()
		self.output_channels = output_channels
		self.hidden_channels = hidden_channels
		self.layer = nn.Sequential(
			ResidualConnect(
				nn.Conv2d(3, hidden_channels[0], (2, 2), (2, 2), bias=False),
				nn.Sequential(
					nn.Conv2d(3, hidden_channels[0], (3, 3), (1, 1), padding=1),
					nn.ReLU(),
					nn.Conv2d(hidden_channels[0], hidden_channels[0], (3, 3), (1, 1), padding=1),
					nn.MaxPool2d((2, 2)))),
			nn.ReLU(),
			ResidualConnect(
				nn.Conv2d(hidden_channels[0], hidden_channels[1], (2, 2), (2, 2), bias=False),
				nn.Sequential(
					nn.Conv2d(hidden_channels[0], hidden_channels[1], (3, 3), (1, 1), padding=1),
					nn.ReLU(),
					nn.Conv2d(hidden_channels[1], hidden_channels[1], (3, 3), (1, 1), padding=1),
					nn.MaxPool2d((2, 2)))),
			nn.ReLU(),
			nn.Conv2d(hidden_channels[1], output_channels, (1, 1), (1, 1))
			)
	def forward(self, x):
		return self.layer(x)


class RecurrentCore(nn.Module):
	def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
		super(RecurrentCore, self).__init__()

		assert hidden_channels % 2 == 0

		self.input_channels = input_channels
		self.hidden_channels = hidden_channels
		self.bias = bias
		self.kernel_size = kernel_size
		self.num_features = 4

		self.padding = [int((ks - 1) / 2) for ks in kernel_size]
		
		input_size = self.input_channels + self.hidden_channels
		hidden_size = 4 * self.hidden_channels
		self.conv_layer = nn.Conv2d(input_size, hidden_size, self.kernel_size, padding=self.padding, bias=self.bias)
		
	def forward(self, x, h, c):
		inp = torch.cat([x, h], dim=1)
		gf, gi, gg, go = torch.chunk(self.conv_layer(inp), 4, dim=1)
		
		gf = torch.sigmoid(gf)
		gi = torch.sigmoid(gi)
		gg = torch.tanh(gg)
		go = torch.sigmoid(go)
		c = gf * c + gi * gg
		h = go * torch.tanh(c)
		
		return h, c


class PosteriorFactor(nn.Module):
	def __init__(self, input_channels, output_channels, hidden_channels=[128, 256]):
		super(PosteriorFactor, self).__init__()
		self.input_channels = input_channels
		self.output_channels = output_channels 
		self.hidden_channels = hidden_channels
		self.layer = nn.Sequential(
			nn.Conv2d(input_channels, hidden_channels[0], (5, 5), (1, 1), padding=2),
			nn.ReLU(),
			ResidualConnect(
				nn.Conv2d(hidden_channels[0], hidden_channels[1], (5, 5), (1, 1), padding=2),
				nn.Conv2d(hidden_channels[0], hidden_channels[1], (1, 1), (1, 1), bias=False)),
			nn.ReLU(),
			ResidualConnect(
				nn.Conv2d(hidden_channels[1], output_channels, (5, 5), (1, 1), padding=2),
				nn.Conv2d(hidden_channels[1], output_channels, (1, 1), (1, 1), bias=False)),
			nn.ReLU(),
			nn.Conv2d(output_channels, 2 * output_channels, (1, 1), (1, 1)),  # mean, log-variance
		)
		
	def forward(self, inp):
		loc, lva = torch.chunk(self.layer(inp), 2, dim=1)
		scale = (0.5 * lva).exp()
		qd = Normal(loc, scale)
		return qd, loc, lva


class GeneratorCoreDelta(nn.Module):
	def __init__(self, input_channels, output_channels, hidden_channels):
		super(GeneratorCoreDelta, self).__init__()
		self.layer = nn.Sequential(
			ResidualConnect(
				nn.Conv2d(input_channels, hidden_channels, (3, 3), (1, 1), padding=1),
				nn.Conv2d(input_channels, hidden_channels, (1, 1), (1, 1), bias=False)),
			nn.ReLU(),
			nn.Conv2d(hidden_channels, output_channels, (3, 3), (1, 1), padding=1)
		)
	
	def forward(self, inp):
		return inp + self.layer(inp)


class GeneratorCoreOutlet(nn.Module):
	def __init__(self, input_channels, hidden_channels=32):
		super(GeneratorCoreOutlet, self).__init__()
		self.layer = nn.Sequential(
			ResidualConnect(
				nn.ConvTranspose2d(input_channels, input_channels, (6, 6), stride=(2, 2), padding=(2, 2), bias=False),
				nn.Sequential(
					nn.ConvTranspose2d(input_channels, hidden_channels, (5, 5), (1, 1), (0, 0)), nn.ReLU(),
					nn.ConvTranspose2d(hidden_channels, hidden_channels, (5, 5), (1, 1), (0, 0)), nn.ReLU(),
					nn.ConvTranspose2d(hidden_channels, hidden_channels, (5, 5), (1, 1), (0, 0)), nn.ReLU(),
					nn.ConvTranspose2d(hidden_channels, input_channels, (5, 5), (1, 1), (0, 0)))
			),
			nn.ReLU(),
			ResidualConnect(
				nn.ConvTranspose2d(input_channels, input_channels, (6, 6), stride=(2, 2), padding=(2, 2), bias=False),
				nn.Sequential(
					nn.ConvTranspose2d( input_channels, hidden_channels, (9, 9), (1, 1), (0, 0)), nn.ReLU(),
					nn.ConvTranspose2d(hidden_channels, hidden_channels, (9, 9), (1, 1), (0, 0)), nn.ReLU(),
					nn.ConvTranspose2d(hidden_channels, hidden_channels, (9, 9), (1, 1), (0, 0)), nn.ReLU(),
					nn.ConvTranspose2d(hidden_channels,  input_channels, (9, 9), (1, 1), (0, 0)))
			),
			nn.ReLU(),
			nn.Conv2d(input_channels, 3, (3, 3), (1, 1), padding=1)
		)
	def forward(self, inp):
		return self.layer(inp)


class QueryNet(nn.Module):
	def __init__(self, r=128, v=7, xi=64, z=64, u=64, hg=64, he=64, bias=True):
		"""
		: channel sizes :
			r  - scene encoder output
			v  - viewpoint, default 7, (X, Y, Z sin(P), cos(P), sin(H), cos(H))
			xi - view inlet output
			z  - variational sample
			u  - generator core output
			hg - generator core hidden
			he - inference core hidden
		"""
		super(QueryNet, self).__init__()
		self.bias = bias
		self.channel_sizes = {
			'r': r, 'v': v, 'xi': xi, 'z': z, 'u': u, 'hg': hg, 'he': he
		}

		self.scene_encoder = PoolSceneEncoder(output_channels=r, hidden_channels=64)
		self.prior_factor = PriorFactor(input_channels=hg, output_channels=z)
		self.icore_inlet = InferenceCoreInlet(output_channels=xi)
		self.icore_cell = RecurrentCore(input_channels=xi+v+r+hg+u, hidden_channels=64, kernel_size=(5, 5))
		self.posterior_factor = PosteriorFactor(input_channels=he, output_channels=z)
		self.gcore_cell = RecurrentCore(input_channels=v+r+z, hidden_channels=64, kernel_size=(5, 5))
		self.gcore_delta = GeneratorCoreDelta(input_channels=hg, output_channels=u, hidden_channels=128)
		self.gcore_outlet = GeneratorCoreOutlet(input_channels=u)


	def forward(self, K, xk, vk, xq, vq, ar=4):
		bs = xq.size(0)
		dev = xk.device
		hg = torch.zeros(bs, self.channel_sizes['hg'], 16, 16, device=dev)
		cg = torch.zeros(bs, self.channel_sizes['hg'], 16, 16, device=dev)
		he = torch.zeros(bs, self.channel_sizes['he'], 16, 16, device=dev)
		ce = torch.zeros(bs, self.channel_sizes['he'], 16, 16, device=dev)
		u = torch.zeros(bs, self.channel_sizes['u'], 16, 16, device=dev)
		
		ploc = []
		plva = []
		qloc = []
		qlva = []
	
		try:
			r  = self.scene_encoder(xk, vk).view(-1, K, self.channel_sizes['r'], 1, 1).sum(1)
		except:
			import pdb
			pdb.set_trace()
		for _ in range(ar):
			pd, pm, pv = self.prior_factor(hg)
			
			iq = self.icore_inlet(xq)
			try:
				iq = torch.cat([iq, vq.expand(-1, -1, 16, 16), r.expand(-1, -1, 16, 16), hg, u], dim=1)
			except:
				import pdb
				pdb.set_trace()
			he, ce = self.icore_cell(iq, he, ce)
			
			qd, qm, qv = self.posterior_factor(he)
			z = qd.rsample()
			
			ig = torch.cat([vq.expand(-1, -1, 16, 16), r.expand(-1, -1, 16, 16), z], dim=1)
			hg, cg = self.gcore_cell(ig, hg, cg)
			
			u = u + self.gcore_delta(hg)
			
			ploc.append(pm)
			plva.append(pv)
			qloc.append(qm)
			qlva.append(qv)
		
		x = self.gcore_outlet(u)
		return x, [qloc, qlva, ploc, plva]


def sample_data(base_path, B, K, N, cuda):
	# B, batch size; K, number of views; N, scnen number [from, to]
	xk = []
	vk = []
	xq = []
	vq = []

	if type(N) is int:
		N = [0, N]
	else:
		assert len(N) == 2

	for b in randint(N[0], N[1], B):
		scene_path = os.path.join(base_path, '{:08d}').format(b)
		npyfile = os.path.join(scene_path, 'xyzhp.npy')
		xyzph = np.load(npyfile)
		for k in randint(0, 5, K):
			imgfile = os.path.join(scene_path, '{:08d}_{:02d}.jpg').format(b, k)
			img = imageio.imread(imgfile).transpose(2, 0, 1) / 255
			xk.append(img)
			vk.append(xyzph[k])
			
		k = randint(0, 5)
		imgfile = os.path.join(scene_path, '{:08d}_{:02d}.jpg').format(b, k)
		img = imageio.imread(imgfile).transpose(2, 0, 1) / 255
		xq.append(img)
		vq.append(xyzph[k])
	
	xk = torch.tensor(xk, dtype=torch.float32, device='cuda' if cuda else 'cpu')
	vk = torch.tensor(vk, dtype=torch.float32, device='cuda' if cuda else 'cpu').view(B * K, 7, 1, 1)
	xq = torch.tensor(xq, dtype=torch.float32, device='cuda' if cuda else 'cpu')
	vq = torch.tensor(vq, dtype=torch.float32, device='cuda' if cuda else 'cpu').view(B, 7, 1, 1)
	
	return K, xk, vk, xq, vq


def kl_divergence(qm, qv, pm, pv):
	return (-pv + qv).exp().sum(1) + (qm - pm).pow(2).div(pv.exp()).sum(1) + pv.sum(1) - qv.sum(1)


def anneal_sigma(epoch, epoch_max, sigma_min=0.7, sigma_max=2.0):
	return max(sigma_min + (sigma_max - sigma_min) * (1 - epoch / epoch_max), sigma_min)


def anneal_lr(optimiser, epoch, n=1.6e6, lr_min=5e-5, lr_max=5e-4):
	lr = max(lr_min + (lr_max - lr_min) * (1 - epoch / n), lr_min)
	for param_group in optimiser.param_groups:
		param_group['lr'] = lr


def train(model, optimiser, base_path, 
		  train_scene_range, test_scene_range,
		  epochs=2e6, batches=32, ar=12, 
		  sigma_min=0.7, sigma_max=2.0, 
		  lr_max=5e-4, lr_min=5e-5, 
		  cuda=True,
		  fhndl=None, save_path='', save_every=10000):
	for t in tqdm(range(epochs)):
		K, xk, vk, xq, vq = sample_data(base_path, batches, randint(1, 6), train_scene_range, cuda)
		x, qpstat = model(K, xk, vk, xq, vq, ar=ar)

		sigma = anneal_sigma(t, epochs, int(.8 * epochs), lr_max, lr_min)
		anneal_lr = anneal_lr(optimiser, t, sigma_min, sigma_max)

		kl = 0
		for m0, v0, m1, v1 in zip(*qpstat):
			kl = kl + kl_divergence(m0, v0, m1, v1).sum(2).sum(1)
		kl = kl.mean()
			
		sqe_train = (x - xq).pow(2).sum(3).sum(2).sum(1).mean()
		
		(1./sigma * sqe_train + kl).backward()
		optimiser.step()
		optimiser.zero_grad()

		if fhndl is not None and os.path.exists(save_path):
			with torch.no_grad():
				K, xk, vk, xq, vq = sample_data(base_path, batches, randint(1, 6), test_scene_range, cuda)
				x, qpstat = model(K, xk, vk, xq, vq, ar=ar)
				sqe_test = (x - xq).pow(2).sum(3).sum(2).sum(1).mean()

			if t == 0 or (t + 1) % save_every == 0:
				torch.save(model.state_dict(), os.path.join(save_path, 'model_{:08d}.pth'.format(t)))
		