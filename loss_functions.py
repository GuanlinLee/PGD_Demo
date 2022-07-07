from torch.autograd import Variable
import torch
import torch.nn.functional as F


def PGD(model, x, y, optimizer, args):
	model.eval()
	epsilon = args.eps
	num_steps = args.ns
	step_size = args.ss
	x_adv = x.detach() + torch.FloatTensor(*x.shape).uniform_(-epsilon, epsilon).cuda()
	for _ in range(num_steps):
		x_adv.requires_grad_()
		with torch.enable_grad():
			f_adv, logits_adv = model(x_adv, True)
			loss = F.cross_entropy(logits_adv, y)
		grad = torch.autograd.grad(loss, [x_adv])[0]
		x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
		x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
		x_adv = torch.clamp(x_adv, 0.0, 1.0)
	model.train()
	x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
	# zero gradient
	optimizer.zero_grad()
	# calculate robust loss
	f_adv, logits = model(x_adv, True)
	loss = F.cross_entropy(logits, y)
	return logits, loss
