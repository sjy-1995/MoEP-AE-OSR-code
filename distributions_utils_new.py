import numpy as np
import torch
from scipy.special import gamma


def sample_gamma(m, v):
	# m = m.cpu().numpy()
	# v = v.cpu().numpy()
	z = torch.distributions.gamma.Gamma(m, v).sample()
	# z = torch.normal(m, v)
	# z = torch.from_numpy(z)
	# z = z.cuda()
	return z


def sample_exponpow(mu, sigmap, p):
	# numpy2ri.activate()
	# samples = np.array(alp.rnormp(mu.cpu().detach().numpy(), sigmap.cpu().detach().numpy(), p.cpu().detach().numpy()))
	samples = alp.rnormp(mu.cpu().detach().numpy(), sigmap.cpu().detach().numpy(), p.cpu().detach().numpy())
	# numpy2ri.deactivate()
	samples = np.array(samples)
	z = torch.Tensor(samples).cuda()
	# z = torch.Tensor(samples)
	return z


def metropolis_hastings_old(target_density, mu, size=500000):
	# burnin_size = 10000
	# burnin_size = 500
	# burnin_size = 2000
	burnin_size = 100
	size += burnin_size
	# x0 = np.array([[0, 0]])
	x0 = np.array([[[0]*mu.shape[1]]*mu.shape[0]])
	xt = x0
	samples = []
	# for i in tqdm(range(size)):
	for i in range(size):
		# xt_candidate = np.array([np.random.multivariate_normal(np.zeros(mu.shape[1]), np.eye(mu.shape[1]))])
		# xt_candidate = np.array([np.random.normal(np.zeros(mu.shape[1]), np.ones(mu.shape[1]))])
		xt_candidate = np.array([np.random.normal(np.zeros((mu.shape[0], mu.shape[1])), np.ones((mu.shape[0], mu.shape[1])))])
		# accept_prob = (target_density(xt_candidate))/(target_density(xt))
		accept_prob = (target_density(xt_candidate[0]))/(target_density(xt[0]))
		# print(accept_prob)
		if np.random.uniform(0, 1) < accept_prob:
			xt = xt_candidate
		samples.append(xt)
	samples = np.array(samples[burnin_size:])
	# print(samples.shape)
	# samples = np.reshape(samples, [samples.shape[0], 2])
	return samples[0, 0, :, :]


def metropolis_hastings(target_density, mu, size=500000):
	burnin_size = 10
	size += burnin_size
	x0 = torch.Tensor(np.array([[[0]*mu.shape[0]]*1])).cuda()
	xt = x0
	samples = []
	for i in range(size):
		xt_candidate = torch.Tensor(np.array([np.random.normal(np.zeros((1, mu.shape[0])), np.ones((1, mu.shape[0])))])).cuda()  # 1, 1, 1
		# print(target_density(xt_candidate[0]))
		# print(target_density(xt[0]))
		accept_prob = (target_density(xt_candidate[0]))/(target_density(xt[0]) + 1e-8)
		# print(accept_prob)
		if np.random.uniform(0, 1) < accept_prob:
			xt = xt_candidate
		samples.append(xt)
	samples = samples[burnin_size:]
	# print(samples.shape)
	return samples[0][0, :, :]


def metropolis_hastings_batchall(target_density, mu, size=500000):
	# mu:[b,n,k]  samples:[b,n]
	burnin_size = 200
	# burnin_size = 2000
	size += burnin_size
	x0 = torch.Tensor(np.array([[0]*mu.shape[1]]*mu.shape[0])).cuda()   # b, n
	xt = x0
	samples = []
	for i in range(size):

		if i == 0:
			# xt_candidate = torch.Tensor(np.array(np.random.normal(np.zeros((mu.shape[0], mu.shape[1])), np.ones((mu.shape[0], mu.shape[1]))))).cuda()  # b, n
			xt_candidate = torch.normal(torch.zeros(mu.shape[0], mu.shape[1]), torch.ones(mu.shape[0], mu.shape[1])).cuda()  # b, n
		else:
			xt_candidate = torch.normal(xt.cpu(), torch.ones(mu.shape[0], mu.shape[1])).cuda()  # b, n

		# print(target_density(xt_candidate[0]))
		# print(target_density(xt[0]))

		# accept_prob = (target_density(xt_candidate[0]))/(target_density(xt[0]) + 1e-8)
		# if np.random.uniform(0, 1) < accept_prob:
		# 	xt = xt_candidate

		accept_prob_matrix = (target_density(xt_candidate))/(target_density(xt) + 1e-8)  # b, n

		# if torch.isnan(accept_prob_matrix).any():
			# print(accept_prob_matrix)
			# accept_prob_matrix = torch.where(torch.isnan(accept_prob_matrix), torch.full_like(accept_prob_matrix, 0), accept_prob_matrix)

		# print(torch.isnan(accept_prob_matrix).int().sum())
		# print(accept_prob_matrix)

		rand_matrix = torch.rand([mu.shape[0], mu.shape[1]]).cuda()  # b, n

		# xt = torch.where(rand_matrix < accept_prob_matrix, xt_candidate, x0)
		xt = torch.where(rand_matrix < accept_prob_matrix, xt_candidate, xt)

		samples.append(xt)
	samples = samples[burnin_size:]
	# print(samples.shape)
	# return samples[0][0, :, :]
	return samples[0]


def posterior(x, mu_dn, sigmap_dn, p_dn, mu_up, sigmap_up, p_up):
	y = 1 / ((2 * sigmap_dn * (p_dn ** (1 / p_dn)) * gamma(1 + 1 / p_dn)) * (2 * sigmap_up * (p_up ** (1 / p_up)) * gamma(1 + 1 / p_up))) * np.exp(- ((np.abs(x - mu_dn) ** p_dn) / (p_dn * (sigmap_dn ** p_dn)) + (np.abs(x - mu_up) ** p_up) / (p_up * (sigmap_up ** p_up))))
	# y = 1 / ((2 * sigmap_dn * (p_dn ** (1 / p_dn)) * gamma(1 + 1 / p_dn)) * (2 * sigmap_up * (p_up ** (1 / p_up)) * gamma(1 + 1 / p_up))) * np.exp(- ((np.linalg.norm(x - mu_dn) ** p_dn) / (p_dn * (sigmap_dn ** p_dn)) + (np.linalg.norm(x - mu_up) ** p_up) / (p_up * (sigmap_up ** p_up))))
	# return y
	return np.linalg.norm(y)


def myself_sample_posterior(mu_dn, sigmap_dn, p_dn, mu_up, sigmap_up, p_up):
	poster = lambda x: posterior(x, mu_dn.detach().cpu().numpy(), sigmap_dn.detach().cpu().numpy(), p_dn.detach().cpu().numpy(), mu_up.detach().cpu().numpy(), sigmap_up.detach().cpu().numpy(), p_up.detach().cpu().numpy())
	# poster = lambda x: np.log(posterior(x, mu_dn.detach().cpu().numpy(), sigmap_dn.detach().cpu().numpy(), p_dn.detach().cpu().numpy(), mu_up.detach().cpu().numpy(), sigmap_up.detach().cpu().numpy(), p_up.detach().cpu().numpy()))
	# poster = lambda x: x * 2
	# print(poster)
	# print(poster(np.ones_like(mu_dn.detach().cpu().numpy())))
	# with pm.Model() as model:
	# 	pm.DensityDist('x', poster)
	# 	step = pm.Metropolis(tune=False)
	# 	# step = pm.NUTS()
	# 	trace = pm.sample(step=step)
	# with model:
	# 	results = trace['x']
	# 	# trance = results
	# 	print(results)
	# 	return results
	samples = metropolis_hastings(poster, mu_dn, size=1)
	y_posteriors = poster(samples)

	# print(samples.shape)
	# print(samples)
	# print(mu_dn.cpu().detach().numpy().shape)
	# print(sigmap_dn.cpu().detach().numpy().shape)
	# print(p_dn.cpu().detach().numpy().shape)
	# print(p_dn)

	# y_priors = np.array(alp.dnormp(samples, mu_dn.cpu().detach().numpy(), sigmap_dn.cpu().detach().numpy(), p_dn.cpu().detach().numpy(), log=False))
	y_priors = np.array(alp.dnormp(samples, mu_dn.cpu().detach().numpy(), sigmap_dn.cpu().detach().numpy(), p_dn.cpu().detach().numpy(), log=True))
	samples = torch.Tensor(samples).cuda()
	# samples = torch.Tensor(samples)
	# kl = np.array(np.linalg.norm(y_posteriors - y_priors))
	kl = np.array(np.linalg.norm(np.log(y_posteriors) - y_priors))
	# print(kl)
	kl = torch.Tensor(kl)
	kl = kl.cuda()
	# kl = kl
	return samples, kl


def kl_normal_gamma(qm, qv, pm, pv, yh):

	# dis = torch.div(yh, pv)
	# yh_ = torch.maximum(yh, torch.zeros(yh.shape).cuda())
	# zero = torch.zeros_like(list(yh)).cuda()
	# yh = torch.where(yh < 0, zero, yh)
	dis = yh / pv
	samples = torch.distributions.gamma.Gamma(pm+dis, pv).sample()
	# p_ = torch.distributions.gamma.Gamma(pm+dis, pv).log_prob(samples).exp().cuda()
	# q_ = torch.distributions.gamma.Gamma(qm, qv).log_prob(samples).exp().cuda()
	p_ = torch.distributions.gamma.Gamma(pm+dis, pv).log_prob(samples).cuda()
	q_ = torch.distributions.gamma.Gamma(qm, qv).log_prob(samples).cuda()
	# p_ = torch.distributions.gamma.Gamma(pm+dis, pv).log_prob(samples)
	# q_ = torch.distributions.gamma.Gamma(qm, qv).log_prob(samples)
	# p_ = torch.normal(pm+dis, pv).cuda()
	# q_ = torch.normal(qm, qv).cuda()
	# print(yh)
	# print(p_)
	# print(q_)
	# print(p_.shape)  # batch_size, n
	# print(q_.shape)  # batch_size, n
	# element_wise = scipy.stats.entropy(p_, q_)
	# element_wise = scipy.stats.entropy(p_, q_, axis=1)
	# element_wise = scipy.stats.wasserstein_distance(p_, q_)
	# element_wise = torch.nn.KLDivLoss(reduction='none')(torch.log(q_), p_)
	# p = torch.Tensor(p_)
	# q = torch.Tensor(q_)
	# element_wise = p_ * torch.log(p_ / q_)
	# print(element_wise.shape)
	kl = torch.norm(p_ - q_)
	# element_wise = torch.Tensor(element_wise)
	# kl = element_wise.sum(-1)
	# kl = torch.from_numpy(kl)
	kl = kl.cuda()
	# kl = kl
	# print(kl)

	return kl


def kl_normal_exponpow_latent(q_mu, q_sigmap, q_p, p_mu, p_sigmap, p_p, yh):

	samples_p = np.array(alp.rnormp((p_mu + yh).cpu().detach().numpy(), p_sigmap.cpu().detach().numpy(), p_p.cpu().detach().numpy()))
	samples_q = np.array(alp.rnormp(q_mu.cpu().detach().numpy(), q_sigmap.cpu().detach().numpy(), q_p.cpu().detach().numpy()))
	# p_ = np.array(alp.dnormp(samples_p, (p_mu + yh).cpu().detach().numpy(), p_sigmap.cpu().detach().numpy(), p_p.cpu().detach().numpy(), log=True))
	# q_ = np.array(alp.dnormp(samples_q, q_mu.cpu().detach().numpy(), q_sigmap.cpu().detach().numpy(), q_p.cpu().detach().numpy(), log=True))
	p_ = np.array(alp.dnormp(samples_p, (p_mu + yh).cpu().detach().numpy(), p_sigmap.cpu().detach().numpy(), p_p.cpu().detach().numpy(), log=False))
	q_ = np.array(alp.dnormp(samples_q, q_mu.cpu().detach().numpy(), q_sigmap.cpu().detach().numpy(), q_p.cpu().detach().numpy(), log=False))
	kl = np.array(np.linalg.norm(p_ - q_))
	kl = torch.Tensor(kl)
	kl = kl.cuda()

	return kl


def function_MMPPD(x, mu, sigmap, p, K):
	# p_repeat = torch.repeat_interleave(p.unsqueeze(1), repeats=64, dim=1)
	p_repeat = p.unsqueeze(1).repeat(1, 64, 1)
	# x = np.expand_dims(x, 2).repeat(4, axis=2)
	x = x.unsqueeze(2).repeat(1, 1, K.shape[1])
	# b, n, k
	# mu = mu.detach().cpu().numpy()
	# sigmap = sigmap.detach().cpu().numpy()
	# p_repeat = p_repeat.detach().cpu().numpy()
	# K = K.detach().cpu().numpy()
	K_f_x = 1 / (2 * sigmap * (p_repeat ** (1 / p_repeat)) * torch.exp(torch.lgamma(1 + 1 / p_repeat)) + 1e-8) * torch.exp(- ((torch.abs(x - mu) ** p_repeat) / (p_repeat * (sigmap ** p_repeat) + 1e-8)))
	# print(K_f_x.shape)
	# print(np.expand_dims(K, axis=1).repeat(64, axis=1).shape)
	# f_x = (K_f_x * (np.expand_dims(K, axis=1).repeat(64, axis=1))).sum(axis=2)
	f_x = torch.sum(K_f_x * (K.unsqueeze(1).repeat(1, 64, 1)), dim=2)
	# f_x_ = np.prod(f_x, axis=1)
	#
	# ###############################
	# f_x_ = np.sum(f_x_)
	# f_x_ = np.linalg.norm(f_x)
	f_x_ = torch.prod(f_x, dim=1)
	# f_x_ = torch.mean(f_x_)
	# print(f_x)
	# f_x_ = torch.norm(f_x)
	f_x_ = torch.norm(f_x_)

	return f_x_


def function_MMPPD_batch(x, mu, sigmap, p, K):
	p_repeat = p.unsqueeze(0).unsqueeze(1).repeat(1, mu.shape[0], 1)
	x = x.unsqueeze(2).repeat(1, 1, K.shape[0])
	mu = mu.unsqueeze(0)
	sigmap = sigmap.unsqueeze(0)
	# b, n, k
	K_f_x = 1 / (2 * sigmap * (p_repeat ** (1 / p_repeat)) * torch.exp(torch.lgamma(1 + 1 / p_repeat)) + 1e-8) * torch.exp(- ((torch.abs(x - mu) ** p_repeat) / (p_repeat * (sigmap ** p_repeat) + 1e-8)))
	f_x = torch.sum(K_f_x * (K.unsqueeze(0).unsqueeze(1).repeat(1, mu.shape[0], 1)), dim=2)
	# ###############################
	f_x_ = torch.prod(f_x, dim=1)
	# print(f_x.shape)

	return f_x_


def function_MMPPD_batchall(x, mu, sigmap, p, K):
	# x:[b,n]   mu:[b,n,k]   sigmap:[b,n,k]   p:[b,k]   K:[b,k]
	p_repeat = p.unsqueeze(1).repeat(1, mu.shape[1], 1)
	x = x.unsqueeze(2).repeat(1, 1, K.shape[1])
	# b, n, k
	K_f_x = 1 / (2 * sigmap * (p_repeat ** (1 / p_repeat)) * torch.exp(torch.lgamma(1 + 1 / p_repeat)) + 1e-8) * torch.exp(- (((torch.abs(x - mu) + 1e-8) ** p_repeat) / (p_repeat * (sigmap ** p_repeat) + 1e-8)))
	# K_f_x = 1 / (2 * sigmap * (torch.sqrt((torch.exp(torch.lgamma(1 / p_repeat))) / (torch.exp(torch.lgamma(3 / p_repeat))))) * torch.exp(torch.lgamma(1 + 1 / p_repeat)) + 1e-8) * torch.exp(- ((torch.abs(x - mu) ** p_repeat) / (p_repeat * (sigmap ** p_repeat) + 1e-8)))
	f_x = torch.sum(K_f_x * (K.unsqueeze(1).repeat(1, mu.shape[1], 1)), dim=2)
	# ###############################
	# f_x_ = torch.prod(f_x, dim=1)
	# print(f_x.shape)

	return f_x


def f_MMPPD(samples, mu, sigmap, p, K):
	# samples is in numpy form
	p_repeat = p.unsqueeze(1).repeat(1, mu.shape[1], 1)
	samples = samples.unsqueeze(2).repeat(1, 1, K.shape[1])
	# b, n, k

	K_f_samples = 1 / (2 * sigmap * (p_repeat ** (1 / p_repeat)) * torch.exp(torch.lgamma(1 + 1 / p_repeat)) + 1e-8) * torch.exp(- (((torch.abs(samples - mu) + 1e-8) ** p_repeat) / (p_repeat * (sigmap ** p_repeat) + 1e-8)))
	# K_f_samples = 1 / (2 * sigmap * (torch.sqrt((torch.exp(torch.lgamma(1 / p_repeat))) / (torch.exp(torch.lgamma(3 / p_repeat))))) * torch.exp(torch.lgamma(1 + 1 / p_repeat)) + 1e-8) * torch.exp(- ((torch.abs(samples - mu) ** p_repeat) / (p_repeat * (sigmap ** p_repeat) + 1e-8)))
	# batch_size, n=64, k
	# f_samples = (K_f_samples * (np.expand_dims(K, axis=1).repeat(64, axis=1))).sum(axis=2)  # batch_size, n=64
	f_samples = torch.sum(K_f_samples * (K.unsqueeze(1).repeat(1, mu.shape[1], 1)), dim=2)  # batch_size, n=64
	# suppose each dim is independent
	# f_samples_ = torch.prod(f_samples, dim=1)  # batch_size, 1
	f_samples_ = f_samples
	# f_samples_ = torch.Tensor(f_samples_)
	# f_samples_ = f_samples_.cuda()

	return f_samples_


def f_MMPPD_20210610(samples, mu, sigmap, p):

	f_samples = 1 / (2 * sigmap * (p ** (1 / p)) * torch.exp(torch.lgamma(1 + 1 / p)) + 1e-8) * torch.exp(- (((torch.abs(samples - mu) + 1e-8) ** p) / (p * (sigmap ** p) + 1e-8)))

	return f_samples


def f_MMPPD_change(samples, mu, sigmap, p, K):
	# samples is in numpy form
	p_repeat = p.unsqueeze(1).repeat(1, 64, 1)
	samples = samples.unsqueeze(2).repeat(1, 1, K.shape[1])
	# b, n, k
	# K_f_samples = 1 / (2 * sigmap * p_repeat) * torch.exp(- ((torch.abs(samples - mu) ** p_repeat) / (sigmap ** p_repeat + 1e-8)))
	K_f_samples = 1 / (2 * sigmap + 1e-8) * torch.exp(- ((torch.abs(samples - mu) ** p_repeat) / (sigmap + 1e-8)))
	# K_f_samples = 1 / (2 * sigmap * (torch.sqrt((torch.exp(torch.lgamma(1 / p_repeat))) / (torch.exp(torch.lgamma(3 / p_repeat))))) * torch.exp(torch.lgamma(1 + 1 / p_repeat)) + 1e-8) * torch.exp(- ((torch.abs(samples - mu) ** p_repeat) / (p_repeat * (sigmap ** p_repeat) + 1e-8)))
	# batch_size, n=64, k
	# f_samples = (K_f_samples * (np.expand_dims(K, axis=1).repeat(64, axis=1))).sum(axis=2)  # batch_size, n=64
	f_samples = torch.sum(K_f_samples * (K.unsqueeze(1).repeat(1, 64, 1)), dim=2)  # batch_size, n=64
	# suppose each dim is independent
	# f_samples_ = np.prod(f_samples, axis=1)  # batch_size, 1
	f_samples_ = f_samples
	# f_samples_ = torch.Tensor(f_samples_)
	# f_samples_ = f_samples_.cuda()

	return f_samples_


def sample_MMPPD(mu, sigmap, p, K):
	# p_repeat = torch.repeat_interleave(p.unsqueeze(1), repeats=64, dim=1)
	# mu = mu.detach().cpu().numpy()
	# sigmap = sigmap.detach().cpu().numpy()
	# p_repeat = p_repeat.detach().cpu().numpy()
	# K = K.detach().cpu().numpy()
	MMPPD = lambda x: function_MMPPD(x, mu, sigmap, p, K)
	samples = metropolis_hastings(MMPPD, mu, size=1)
	# print(samples)

	return samples


def sample_MMPPD_batch(mu, sigmap, p, K):
	# p_repeat = torch.repeat_interleave(p.unsqueeze(1), repeats=64, dim=1)
	MMPPD = lambda x: function_MMPPD_batch(x, mu, sigmap, p, K)
	samples = metropolis_hastings(MMPPD, mu, size=1)
	# print(samples)

	return samples


def sample_MMPPD_batchall(mu, sigmap, p, K):
	# view the whole batch as entirety
	MMPPD = lambda x: function_MMPPD_batchall(x, mu, sigmap, p, K)
	samples = metropolis_hastings_batchall(MMPPD, mu, size=1)
	# print(samples)

	return samples


def newsample_batchall(mu, sigmap, p, K_choose_one_hot):

	mu = torch.sum(mu * K_choose_one_hot, dim=2)   # b, n
	sigmap = torch.sum(sigmap * K_choose_one_hot, dim=2)  # b, n
	p = p.unsqueeze(1).repeat(1, mu.shape[1], 1)  # b, n, k
	p = torch.sum(p * K_choose_one_hot, dim=2)  # b, n
	gamma_dis = torch.distributions.gamma.Gamma(1 + 1 / p, 2 ** (- p / 2))
	sample_gamma_ranvar = gamma_dis.sample()  # b, n
	delta = (sigmap * (p ** (1 / p)) * (sample_gamma_ranvar ** (1 / p))) / torch.sqrt(2 * torch.ones_like(p))
	sample_rand_unit = torch.rand(mu.shape[0], mu.shape[1]).cuda()  # b, n
	samples = mu - delta + 2 * delta * sample_rand_unit  # b, n

	return samples


def newsample_batchall_try20210622(mu, sigmap, p, K_choose_one_hot):

	mu = torch.sum(mu * K_choose_one_hot, dim=2)   # b, n
	sigmap = torch.sum(sigmap * K_choose_one_hot, dim=2)  # b, n
	# p = p.unsqueeze(1).repeat(1, mu.shape[1], 1)  # b, n, k
	p = torch.sum(p * K_choose_one_hot, dim=2)  # b, n
	gamma_dis = torch.distributions.gamma.Gamma(1 + 1 / p, 2 ** (- p / 2))
	sample_gamma_ranvar = gamma_dis.sample()  # b, n
	delta = (sigmap * (p ** (1 / p)) * (sample_gamma_ranvar ** (1 / p))) / torch.sqrt(2 * torch.ones_like(p))
	sample_rand_unit = torch.rand(mu.shape[0], mu.shape[1]).cuda()  # b, n
	samples = mu - delta + 2 * delta * sample_rand_unit  # b, n

	return samples


def newsample_batchall_20210610(mu, sigmap, p):

	gamma_dis = torch.distributions.gamma.Gamma(1 + 1 / p, 2 ** (- p / 2))
	sample_gamma_ranvar = gamma_dis.sample()  # b', n
	delta = (sigmap * (p ** (1 / p)) * (sample_gamma_ranvar ** (1 / p))) / torch.sqrt(2 * torch.ones_like(p))
	sample_rand_unit = torch.rand(mu.shape[0], mu.shape[1]).cuda()  # b', n
	samples = mu - delta + 2 * delta * sample_rand_unit  # b', n

	# # try to calculate the KL divergence
	# kl_loss = torch.distributions.kl_divergence(2*gamma_dis, 2*gamma_dis)
	# print('The KL-divergence is : ', kl_loss)

	return samples


def newsample_batchall2(mu, sigmap, p):

	mu = torch.sum(mu * K_choose_one_hot, dim=2)   # b, n
	sigmap = torch.sum(sigmap * K_choose_one_hot, dim=2)  # b, n
	p = p.unsqueeze(1).repeat(1, mu.shape[1], 1)  # b, n, k
	p = torch.sum(p * K_choose_one_hot, dim=2)  # b, n
	gamma_dis = torch.distributions.gamma.Gamma(1 + 1 / p, 2 ** (- p / 2))
	sample_gamma_ranvar = gamma_dis.sample()  # b, n
	delta = (sigmap * (p ** (1 / p)) * (sample_gamma_ranvar ** (1 / p))) / torch.sqrt(2 * torch.ones_like(p))
	sample_rand_unit = torch.rand(mu.shape[0], mu.shape[1]).cuda()  # b, n
	samples = mu - delta + 2 * delta * sample_rand_unit  # b, n

	return samples


# def newsample_batchall(mu, sigmap, p, K_choose_one_hot):
#
# 	mu = torch.sum(mu * K_choose_one_hot, dim=2)   # b, n
# 	sigmap = torch.sum(sigmap * K_choose_one_hot, dim=2)  # b, n
# 	p = torch.sum(p * K_choose_one_hot, dim=2)  # b, n
# 	gamma_dis = torch.distributions.gamma.Gamma(1 + 1 / p, 2 ** (- p / 2))
# 	sample_gamma_ranvar = gamma_dis.sample()  # b, n
# 	delta = (sigmap * (p ** (1 / p)) * (sample_gamma_ranvar ** (1 / p))) / torch.sqrt(2 * torch.ones_like(p))
# 	sample_rand_unit = torch.rand(mu.shape[0], mu.shape[1]).cuda()  # b, n
# 	samples = mu - delta + 2 * delta * sample_rand_unit  # b, n
#
# 	return samples


def newsample_newtrain_batchall(mu, sigmap, p):

	mu = mu.permute(0, 2, 1).contiguous().view(-1, 48 * 3)   # b*k, n
	sigmap = sigmap.permute(0, 2, 1).contiguous().view(-1, 48 * 3)   # b*k, n
	p = p.permute(0, 2, 1).contiguous().view(-1, 48 * 3)   # b*k, n

	gamma_dis = torch.distributions.gamma.Gamma(1 + 1 / p, 2 ** (- p / 2))
	sample_gamma_ranvar = gamma_dis.sample()  # b*k, n
	delta = (sigmap * (p ** (1 / p)) * (sample_gamma_ranvar ** (1 / p))) / torch.sqrt(2 * torch.ones_like(p))
	sample_rand_unit = torch.rand(mu.shape[0], mu.shape[1]).cuda()  # b*k, n
	samples = mu - delta + 2 * delta * sample_rand_unit  # b*k, n

	return samples


def sample_weibull_batchall(x_lambda, x_k):

	x_lambda = torch.squeeze(x_lambda)   # b, n
	x_k = torch.squeeze(x_k)   # b, n
	m_weibull = torch.distributions.weibull.Weibull(x_lambda, x_k)
	samples = m_weibull.sample()  # b, n

	return samples


def sample_lognormal_batchall(x_mean, x_sigma):

	x_mean = torch.squeeze(x_mean)   # b, n
	x_sigma = torch.squeeze(x_sigma)   # b, n
	m_lognormal = torch.distributions.log_normal.LogNormal(x_mean, x_sigma)
	samples = m_lognormal.sample()  # b, n

	return samples


def sample_normal_batchall(x_mean2, x_sigma2):

	x_mean2 = torch.squeeze(x_mean2)   # b, n
	x_sigma2 = torch.squeeze(x_sigma2)   # b, n
	m_normal = torch.distributions.normal.Normal(x_mean2, x_sigma2)
	samples = m_normal.sample()  # b, n

	return samples


def sample_gamma_batchall(x_shape, x_rate):

	x_shape = torch.squeeze(x_shape)   # b, n
	x_rate = torch.squeeze(x_rate)   # b, n
	m_gamma = torch.distributions.gamma.Gamma(x_shape, x_rate)
	samples = m_gamma.sample()  # b, n

	return samples


#####################################################################################################


# def sample_gaussian(m, v):
# 	sample = torch.randn(m.shape).cuda()
# 	z = m + (v**0.5)*sample
# 	return z


# def sample_gamma(m, v):
# 	# m = m.cpu().numpy()
# 	# v = v.cpu().numpy()
# 	z = np.random.gamma(m, v)
# 	z = torch.from_numpy(z)
# 	# z = z.cuda()
# 	return z


# def gaussian_parameters(h, dim=-1):
# 	m, h = torch.split(h, h.size(dim) // 2, dim=dim)
# 	v = F.softplus(h) + 1e-8
# 	return m, v


# def kl_normal(qm, qv, pm, pv, yh):
# 	element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm - yh).pow(2) / pv - 1)
# 	kl = element_wise.sum(-1)
# 	#print("log var1", qv)
# 	return kl


# def kl_normal(qm, qv, pm, pv, yh):
#
# 	dis = torch.div(yh, pv)
# 	p_ = np.random.gamma(pm+dis, pv)
# 	q_ = np.random.gamma(qm, qv)
# 	# element_wise = scipy.stats.entropy(p_, q_)
# 	p = torch.Tensor(p_)
# 	q = torch.Tensor(q_)
# 	element_wise = p * torch.log(p / q)
# 	kl = element_wise.sum(-1)
# 	# kl = torch.from_numpy(kl)
# 	# kl = kl.cuda()
#
# 	return kl


