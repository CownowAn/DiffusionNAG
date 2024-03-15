"""Various sampling methods."""

import functools

import torch
import numpy as np
import abc
import sys
import os

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn

from scipy import integrate
from torchdiffeq import odeint
import sde_lib
from models import utils as mutils
from tqdm import trange

from datasets_nas import MetaTestDataset
# from configs.ckpt import META_DATAROOT_NB201, META_DATAROOT_OFA
from all_path import PROCESSED_DATA_PATH

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f'Already registered predictor with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f'Already registered corrector with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(
        config, sde, shape, inverse_scaler, eps, data, conditional=False,
        p=1, prod_w=False, weight_ratio_abs=False, 
        is_meta=False, data_name='cifar10', num_sample=20, is_multi_obj=False):
    """Create a sampling function.

    Args:
        config: A `ml_collections.ConfigDict` object that contains all configuration information.
        sde: A `sde_lib.SDE` object that represents the forward SDE.
        shape: A sequence of integers representing the expected shape of a single sample.
        inverse_scaler: The inverse data normalizer function.
        eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    Returns:
        A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
    """

    sampler_name = config.sampling.method
    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == 'ode':
        sampling_fn = get_ode_sampler(sde=sde,
                                      shape=shape,
                                      inverse_scaler=inverse_scaler,
                                      denoise=config.sampling.noise_removal,
                                      eps=eps,
                                      rtol=config.sampling.rtol,
                                      atol=config.sampling.atol,
                                      device=config.device)
    elif sampler_name.lower() == 'diffeq':
        sampling_fn = get_diffeq_sampler(sde=sde,
                                         shape=shape,
                                         inverse_scaler=inverse_scaler,
                                         denoise=config.sampling.noise_removal,
                                         eps=eps,
                                         rtol=config.sampling.rtol,
                                         atol=config.sampling.atol,
                                         step_size=config.sampling.ode_step,
                                         method=config.sampling.ode_method,
                                         device=config.device)
    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == 'pc':
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        # print(config.sampling.predictor.lower(), config.sampling.corrector.lower())
        if data in ['NASBench201', 'ofa']:
            if is_meta:
                sampling_fn = get_pc_conditional_sampler_meta_nas(sde=sde,
                                                            shape=shape,
                                                            predictor=predictor,
                                                            corrector=corrector,
                                                            inverse_scaler=inverse_scaler,
                                                            snr=config.sampling.snr,
                                                            n_steps=config.sampling.n_steps_each,
                                                            probability_flow=config.sampling.probability_flow,
                                                            continuous=config.training.continuous,
                                                            denoise=config.sampling.noise_removal,
                                                            eps=eps,
                                                            device=config.device,
                                                            regress=config.sampling.regress,
                                                            labels=config.sampling.labels,
                                                            classifier_scale=config.sampling.classifier_scale,
                                                            weight_scheduling=config.sampling.weight_scheduling,
                                                            weight_ratio=config.sampling.weight_ratio,
                                                            t_spot=config.sampling.t_spot,
                                                            t_spot_end=config.sampling.t_spot_end, 
                                                            p=p,
                                                            prod_w=prod_w,
                                                            weight_ratio_abs=weight_ratio_abs,
                                                            data_name=data_name,
                                                            num_sample=num_sample,
                                                            search_space=config.data.name)
            elif is_multi_obj:
                sampling_fn = get_pc_conditional_sampler_nas(sde=sde,
                                                    shape=shape,
                                                    predictor=predictor,
                                                    corrector=corrector,
                                                    inverse_scaler=inverse_scaler,
                                                    snr=config.sampling.snr,
                                                    n_steps=config.sampling.n_steps_each,
                                                    probability_flow=config.sampling.probability_flow,
                                                    continuous=config.training.continuous,
                                                    denoise=config.sampling.noise_removal,
                                                    eps=eps,
                                                    device=config.device,
                                                    regress=config.sampling.regress,
                                                    labels=config.sampling.labels,
                                                    classifier_scale=config.sampling.classifier_scale,
                                                    weight_scheduling=config.sampling.weight_scheduling,
                                                    weight_ratio=config.sampling.weight_ratio,
                                                    t_spot=config.sampling.t_spot,
                                                    t_spot_end=config.sampling.t_spot_end, 
                                                    p=p,
                                                    prod_w=prod_w,
                                                    weight_ratio_abs=weight_ratio_abs)            
            elif conditional:
                sampling_fn = get_pc_conditional_sampler_nas(sde=sde,
                                                    shape=shape,
                                                    predictor=predictor,
                                                    corrector=corrector,
                                                    inverse_scaler=inverse_scaler,
                                                    snr=config.sampling.snr,
                                                    n_steps=config.sampling.n_steps_each,
                                                    probability_flow=config.sampling.probability_flow,
                                                    continuous=config.training.continuous,
                                                    denoise=config.sampling.noise_removal,
                                                    eps=eps,
                                                    device=config.device,
                                                    regress=config.sampling.regress,
                                                    labels=config.sampling.labels,
                                                    classifier_scale=config.sampling.classifier_scale,
                                                    weight_scheduling=config.sampling.weight_scheduling,
                                                    weight_ratio=config.sampling.weight_ratio,
                                                    t_spot=config.sampling.t_spot,
                                                    t_spot_end=config.sampling.t_spot_end, 
                                                    p=p,
                                                    prod_w=prod_w,
                                                    weight_ratio_abs=weight_ratio_abs)
            else:
                sampling_fn = get_pc_sampler_nas(sde=sde,
                                                shape=shape,
                                                predictor=predictor,
                                                corrector=corrector,
                                                inverse_scaler=inverse_scaler,
                                                snr=config.sampling.snr,
                                                n_steps=config.sampling.n_steps_each,
                                                probability_flow=config.sampling.probability_flow,
                                                continuous=config.training.continuous,
                                                denoise=config.sampling.noise_removal,
                                                eps=eps,
                                                device=config.device)                
                
        else:
            sampling_fn = get_pc_sampler(sde=sde,
                                        shape=shape,
                                        predictor=predictor,
                                        corrector=corrector,
                                        inverse_scaler=inverse_scaler,
                                        snr=config.sampling.snr,
                                        n_steps=config.sampling.n_steps_each,
                                        probability_flow=config.sampling.probability_flow,
                                        continuous=config.training.continuous,
                                        denoise=config.sampling.noise_removal,
                                        eps=eps,
                                        device=config.device)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        if isinstance(sde, tuple):
            self.rsde = (sde[0].reverse(score_fn, probability_flow), sde[1].reverse(score_fn, probability_flow))
        else:
            self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t, *args, **kwargs):
        """One update of the predictor.

        Args:
            x: A PyTorch tensor representing the current state.
            t: A PyTorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t, *args, **kwargs):
        """One update of the corrector.

        Args:
            x: A PyTorch tensor representing the current state.
            t: A PyTorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    # def update_fn(self, x, t, *args, **kwargs):
    #     dt = -1. / self.rsde.N
    #     z = torch.randn_like(x)
    #     z = torch.tril(z, -1)
    #     z = z + z.transpose(-1, -2)
    #     drift, diffusion = self.rsde.sde(x, t, *args, **kwargs)
    #     drift = torch.tril(drift, -1)
    #     drift = drift + drift.transpose(-1, -2)
    #     x_mean = x + drift * dt
    #     x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
    #     return x, x_mean
    
    def update_fn(self, x, t, *args, **kwargs):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        # z = torch.tril(z, -1)
        # z = z + z.transpose(-1, -2)
        drift, diffusion = self.rsde.sde(x, t, *args, **kwargs)
        # drift = torch.tril(drift, -1)
        # drift = drift + drift.transpose(-1, -2)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
        return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    # def update_fn(self, x, t, *args, **kwargs):
    #     f, G = self.rsde.discretize(x, t, *args, **kwargs)
    #     f = torch.tril(f, -1)
    #     f = f + f.transpose(-1, -2)
    #     z = torch.randn_like(x)
    #     z = torch.tril(z, -1)
    #     z = z + z.transpose(-1, -2)

    #     x_mean = x - f
    #     x = x_mean + G[:, None, None, None] * z
    #     return x, x_mean
    
    def update_fn(self, x, t, *args, **kwargs):
        f, G = self.rsde.discretize(x, t, *args, **kwargs)
        # f = torch.tril(f, -1)
        # f = f + f.transpose(-1, -2)
        z = torch.randn_like(x)
        # z = torch.tril(z, -1)
        # z = z + z.transpose(-1, -2)

        x_mean = x - f
        x = x_mean + G[:, None, None] * z
        return x, x_mean


@register_predictor(name='none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t, *args, **kwargs):
        return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)

    # def update_fn(self, x, t, *args, **kwargs):
    #     sde = self.sde
    #     score_fn = self.score_fn
    #     n_steps = self.n_steps
    #     target_snr = self.snr
    #     if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
    #         timestep = (t * (sde.N - 1) / sde.T).long()
    #         # Note: it seems that subVPSDE doesn't set alphas
    #         alpha = sde.alphas.to(t.device)[timestep]
    #     else:
    #         alpha = torch.ones_like(t)

    #     for i in range(n_steps):

    #         grad = score_fn(x, t, *args, **kwargs)
    #         noise = torch.randn_like(x)

    #         noise = torch.tril(noise, -1)
    #         noise = noise + noise.transpose(-1, -2)

    #         mask = kwargs['mask']

    #         # mask invalid elements and calculate norm
    #         mask_tmp = mask.reshape(mask.shape[0], -1)

    #         grad_norm = torch.norm(mask_tmp * grad.reshape(grad.shape[0], -1), dim=-1).mean()
    #         noise_norm = torch.norm(mask_tmp * noise.reshape(noise.shape[0], -1), dim=-1).mean()

    #         step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
    #         x_mean = x + step_size[:, None, None, None] * grad
    #         x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    #     return x, x_mean

    def update_fn(self, x, t, *args, **kwargs):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            # Note: it seems that subVPSDE doesn't set alphas
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):

            grad = score_fn(x, t, *args, **kwargs)
            noise = torch.randn_like(x)

            # noise = torch.tril(noise, -1)
            # noise = noise + noise.transpose(-1, -2)

            # mask = kwargs['maskX']

            # mask invalid elements and calculate norm
            # mask_tmp = mask.reshape(mask.shape[0], -1)

            # grad_norm = torch.norm(mask_tmp * grad.reshape(grad.shape[0], -1), dim=-1).mean()
            # noise_norm = torch.norm(mask_tmp * noise.reshape(noise.shape[0], -1), dim=-1).mean()
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()

            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise

        return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t, *args, **kwargs):
        return x, x


def shared_predictor_update_fn(x, t, sde, model, 
                               predictor, probability_flow, continuous, *args, **kwargs):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)

    return predictor_obj.update_fn(x, t, *args, **kwargs)


def shared_corrector_update_fn(x, t, sde, model, 
                               corrector, continuous, snr, n_steps, *args, **kwargs):
    """A wrapper that configures and returns the update function of correctors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)

    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)

    return corrector_obj.update_fn(x, t, *args, **kwargs)


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
        sde: An `sde_lib.SDE` object representing the forward SDE.
        shape: A sequence of integers. The expected shape of a single sample.
        predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
        corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
        inverse_scaler: The inverse data normalizer.
        snr: A `float` number. The signal-to-noise ratio for configuring correctors.
        n_steps: An integer. The number of corrector steps per predictor update.
        probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
        continuous: `True` indicates that the score model was continuously trained.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        device: PyTorch device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def pc_sampler(model, n_nodes_pmf):
        """The PC sampler function.

        Args:
            model: A score model.
            n_nodes_pmf: Probability mass function of graph nodes.

        Returns:
            Samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            # Sample the number of nodes
            n_nodes = torch.multinomial(n_nodes_pmf, shape[0], replacement=True)
            mask = torch.zeros((shape[0], shape[-1]), device=device)
            for i in range(shape[0]):
                mask[i][:n_nodes[i]] = 1.
            mask = (mask[:, None, :] * mask[:, :, None]).unsqueeze(1)
            mask = torch.tril(mask, -1)
            mask = mask + mask.transpose(-1, -2)

            x = x * mask

            for i in range(sde.N):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = corrector_update_fn(x, vec_t, model=model, mask=mask)
                x = x * mask
                x, x_mean = predictor_update_fn(x, vec_t, model=model, mask=mask)
                x = x * mask

            return inverse_scaler(x_mean if denoise else x) * mask, sde.N * (n_steps + 1), n_nodes

    return pc_sampler

def get_pc_sampler_nas(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
        sde: An `sde_lib.SDE` object representing the forward SDE.
        shape: A sequence of integers. The expected shape of a single sample.
        predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
        corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
        inverse_scaler: The inverse data normalizer.
        snr: A `float` number. The signal-to-noise ratio for configuring correctors.
        n_steps: An integer. The number of corrector steps per predictor update.
        probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
        continuous: `True` indicates that the score model was continuously trained.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        device: PyTorch device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def pc_sampler(model, mask):
        """The PC sampler function.

        Args:
            model: A score model.
            n_nodes_pmf: Probability mass function of graph nodes.

        Returns:
            Samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            # Sample the number of nodes
            # n_nodes = torch.multinomial(n_nodes_pmf, shape[0], replacement=True)
            # mask = torch.zeros((shape[0], shape[-1]), device=device)
            # for i in range(shape[0]):
            #     mask[i][:n_nodes[i]] = 1.
            # mask = (mask[:, None, :] * mask[:, :, None]).unsqueeze(1)
            # mask = torch.tril(mask, -1)
            # mask = mask + mask.transpose(-1, -2)
            # x = x * mask
            mask = mask[0].unsqueeze(0).repeat(x.size(0), 1, 1)

            for i in trange(sde.N, desc='[PC sampling]', position=1, leave=False):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = corrector_update_fn(x, vec_t, model=model, maskX=mask)
                # x = x * mask
                x, x_mean = predictor_update_fn(x, vec_t, model=model, maskX=mask)
                # x = x * mask

            # return inverse_scaler(x_mean if denoise else x) * mask, sde.N * (n_steps + 1), n_nodes
            return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1), None

    return pc_sampler


def get_pc_conditional_sampler_nas(sde, shape,
                               predictor, corrector, inverse_scaler, snr,
                               n_steps=1, probability_flow=False,
                               continuous=False, denoise=True, eps=1e-5, device='cuda',
                               regress=True, labels='max', classifier_scale=0.5,
                               weight_scheduling=True, weight_ratio=True, t_spot=1., t_spot_end=None,
                               p=1, prod_w=False, weight_ratio_abs=False):
    """Class-conditional sampling with Predictor-Corrector (PC) samplers.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      score_model: A `torch.nn.Module` object that represents the architecture of the score-based model.
      classifier: A `torch.nn.Module` object that represents the architecture of the noise-dependent classifier.
    #   classifier_params: A dictionary that contains the weights of the classifier.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.predictor` that represents a predictor algorithm.
      corrector: A subclass of `sampling.corrector` that represents a corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for correctors.
      n_steps: An integer. The number of corrector steps per update of the predictor.
      probability_flow: If `True`, solve the probability flow ODE for sampling with the predictor.
      continuous: `True` indicates the score-based model was trained with continuous time.
      denoise: If `True`, add one-step denoising to final samples.
      eps: A `float` number. The SDE/ODE will be integrated to `eps` to avoid numerical issues.

    Returns: A pmapped class-conditional image sampler.
    """
    score_grad_norm_p, classifier_grad_norm_p = [], []
    score_grad_norm_c, classifier_grad_norm_c = [], []
    if t_spot_end is None or t_spot_end == 0.:
        t_spot_end = eps
    
    def weight_scheduling_fn(w, t):
        return w * 0.1 ** t

    def conditional_predictor_update_fn(score_model, classifier, x, t, labels, maskX, *args, **kwargs):
        """The predictor update function for class-conditional sampling."""
        score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous=continuous)
        # The gradient function of the noise-dependent classifier
        classifier_grad_fn = mutils.get_classifier_grad_fn(sde, classifier, train=False, continuous=continuous, 
                                                           regress=regress, labels=labels)

        def total_grad_fn(x, t, *args, **kwargs):
            
            # score =  score_fn(x, t, *args, **kwargs)
            score =  score_fn(x, t, maskX)
            classifier_grad = classifier_grad_fn(x, t, maskX, *args, **kwargs)
            
            # Sample weight
            if weight_scheduling:
                w = weight_scheduling_fn(classifier_scale, t[0].item())
            else:
                w = classifier_scale
            
            if weight_ratio:
                if prod_w:
                    ratio = score.view(x.shape[0], -1).norm(p=p, dim=-1) / (w * classifier_grad).view(x.shape[0], -1).norm(p=p, dim=-1)
                else:
                    ratio = score.view(x.shape[0], -1).norm(p=p, dim=-1) / classifier_grad.view(x.shape[0], -1).norm(p=p, dim=-1)
                # ratio = score.view(x.shape[0], -1).norm(p=p, dim=-1) / classifier_grad.view(x.shape[0], -1).norm(p=p, dim=-1)
                w *= ratio[:, None, None]
            
            if weight_ratio_abs:
                assert not weight_ratio
                ratio = torch.div(torch.abs(score), torch.abs(classifier_grad))
                w *= ratio
            
            score_grad_norm_p.append(torch.mean(score.view(x.shape[0], -1).norm(p=p, dim=-1)).item())
            
            if weight_ratio: # ratio per sample
                classifier_grad_norm_p.append(torch.mean(classifier_grad.view(x.shape[0], -1).norm(p=p, dim=-1) * ratio[:, None, None]).item())
            elif weight_ratio_abs: # ratio per element
                classifier_grad_norm_p.append(torch.mean((classifier_grad * ratio).norm(p=p)).item())
            else:
                classifier_grad_norm_p.append(torch.mean(classifier_grad.view(x.shape[0], -1).norm(p=p, dim=-1)).item())

            if t_spot < 1.:
                if t[0].item() <= t_spot and t[0] >= t_spot_end:
                    return score + w * classifier_grad
                    # return (1 - w) * score + w * classifier_grad
                else:
                    return score
            else:
                # return (1 - w) * score + w * classifier_grad
                return score + w * classifier_grad

        if predictor is None:
            predictor_obj = NonePredictor(sde, total_grad_fn, probability_flow)
        else:
            predictor_obj = predictor(sde, total_grad_fn, probability_flow)
        return predictor_obj.update_fn(x, t, *args, **kwargs)

    def conditional_corrector_update_fn(score_model, classifier, x, t, labels, maskX, *args, **kwargs):
        """The corrector update function for class-conditional sampling."""
        score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous=continuous)
        classifier_grad_fn = mutils.get_classifier_grad_fn(sde, classifier, train=False, continuous=continuous, 
                                                           regress=regress, labels=labels)

        def total_grad_fn(x, t, *args, **kwargs):
            # score =  score_fn(x, t, *args, **kwargs)
            score =  score_fn(x, t, maskX)
            classifier_grad = classifier_grad_fn(x, t, maskX, *args, **kwargs)
            
            # Sample weight
            if weight_scheduling:
                w = weight_scheduling_fn(classifier_scale, t[0].item())
            else:
                w = classifier_scale

            if weight_ratio:
                if prod_w:
                    ratio = score.view(x.shape[0], -1).norm(p=p, dim=-1) / (w * classifier_grad).view(x.shape[0], -1).norm(p=p, dim=-1)
                else:
                    ratio = score.view(x.shape[0], -1).norm(p=p, dim=-1) / classifier_grad.view(x.shape[0], -1).norm(p=p, dim=-1)
                # ratio = score.view(x.shape[0], -1).norm(p=p, dim=-1) / classifier_grad.view(x.shape[0], -1).norm(p=p, dim=-1)
                w *= ratio[:, None, None]

            score_grad_norm_c.append(torch.mean(score.view(x.shape[0], -1).norm(p=p, dim=-1)).item())
            
            if weight_ratio:
                classifier_grad_norm_c.append(torch.mean(classifier_grad.view(x.shape[0], -1).norm(p=p, dim=-1) * ratio[:, None, None]).item())
            else:
                classifier_grad_norm_c.append(torch.mean(classifier_grad.view(x.shape[0], -1).norm(p=p, dim=-1)).item())
            
            if t_spot < 1.:
                if t[0].item() <= t_spot and t[0] >= t_spot_end:
                    return score + w * classifier_grad
                    # return (1 - w) * score + w * classifier_grad
                else:
                    return score
            else:
                return score + w * classifier_grad
                # return (1 - w) * score + w * classifier_grad

        if corrector is None:
            corrector_obj = NoneCorrector(sde, total_grad_fn, snr, n_steps)
        else:
            corrector_obj = corrector(sde, total_grad_fn, snr, n_steps)
        return corrector_obj.update_fn(x, t, *args, **kwargs)

    def pc_conditional_sampler(score_model, mask, classifier, 
                               eval_chain=False, keep_chain=None, number_chain_steps=None):
        """Generate class-conditional samples with Predictor-Corrector (PC) samplers.

        Args:
          score_model: A `torch.nn.Module` object that represents the training state
            of the score-based model.
          labels: A JAX array of integers that represent the target label of each sample.

        Returns:
          Class-conditional samples.
        """
        chain_x = None
        if eval_chain:
            if number_chain_steps is None:
                number_chain_steps = sde.N
            if keep_chain is None:
                keep_chain = shape[0] # all sample
            assert number_chain_steps <= sde.N
            chain_x_size = torch.Size((number_chain_steps, keep_chain, *shape[1:]))
            chain_x = torch.zeros(chain_x_size)
        
        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
            
            if len(mask.shape) == 3:
                mask = mask[0]
            mask = mask.unsqueeze(0).repeat(x.size(0), 1, 1) # adj
            
            for i in trange(sde.N, desc='[PC conditional sampling]', position=1, leave=False):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                # x, x_mean = conditional_corrector_update_fn(x, vec_t, model=model, maskX=mask)
                x, x_mean = conditional_corrector_update_fn(score_model, classifier, x, vec_t, labels=labels, maskX=mask)
                # x = x * mask
                x, x_mean = conditional_predictor_update_fn(score_model, classifier, x, vec_t, labels=labels, maskX=mask)
                # x = x * mask
                
                if eval_chain:
                #     arch_metric = sampling_metrics(arch_list=inverse_scaler(x_mean if denoise else x), 
                #                                    adj=adj, mask=mask, 
                #                                    this_sample_dir=os.path.join(sampling_metrics.exp_name),
                                                #    test=False)
                    # r_valid, r_unique, r_novel = arch_metric[0][0], arch_metric[0][1],  arch_metric[0][2]
                # Save the first keep_chain graphs
                    write_index = number_chain_steps - 1 - int((i * number_chain_steps) // sde.N)
                    # write_index = int((t * number_chain_steps) // sde.T)
                    chain_x[write_index] = inverse_scaler(x_mean if denoise else x)[:keep_chain]
            
            # Overwrite last frame with the resulting x
            # if keep_chain > 0:
            #     final_x_chain = inverse_scaler(x_mean if denoise else x)[:keep_chain]
            #     chain_x[0] = final_x_chain 
            #     # Repeat last frame to see final sample better
            #     import pdb; pdb.set_trace()
            #     chain_x = torch.cat([chain_x, chain_x[-1:].repeat(10, 1, 1)], dim=0)
            #     import pdb; pdb.set_trace()
            #     assert chain_x.size(0) == (number_chain_steps + 10)
            
            return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1), chain_x, (score_grad_norm_p, classifier_grad_norm_p, score_grad_norm_c, classifier_grad_norm_c)
            
    return pc_conditional_sampler


def get_pc_conditional_sampler_meta_nas(sde, shape,
                               predictor, corrector, inverse_scaler, snr,
                               n_steps=1, probability_flow=False,
                               continuous=False, denoise=True, eps=1e-5, device='cuda',
                               regress=True, labels='max', classifier_scale=0.5,
                               weight_scheduling=True, weight_ratio=True, t_spot=1., t_spot_end=None,
                               p=1, prod_w=False, weight_ratio_abs=False,
                               data_name='cifar10', num_sample=20, search_space=None):
    """Class-conditional sampling with Predictor-Corrector (PC) samplers.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      score_model: A `torch.nn.Module` object that represents the architecture of the score-based model.
      classifier: A `torch.nn.Module` object that represents the architecture of the noise-dependent classifier.
    #   classifier_params: A dictionary that contains the weights of the classifier.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.predictor` that represents a predictor algorithm.
      corrector: A subclass of `sampling.corrector` that represents a corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for correctors.
      n_steps: An integer. The number of corrector steps per update of the predictor.
      probability_flow: If `True`, solve the probability flow ODE for sampling with the predictor.
      continuous: `True` indicates the score-based model was trained with continuous time.
      denoise: If `True`, add one-step denoising to final samples.
      eps: A `float` number. The SDE/ODE will be integrated to `eps` to avoid numerical issues.

    Returns: A pmapped class-conditional image sampler.
    """
    
    # --------- Meta-NAS (START) ---------- #
    test_dataset = MetaTestDataset(
        data_path=PROCESSED_DATA_PATH,
        data_name=data_name,
        num_sample=num_sample
    )
    # --------- Meta-NAS (END) ---------- #
    
    score_grad_norm_p, classifier_grad_norm_p = [], []
    score_grad_norm_c, classifier_grad_norm_c = [], []
    if t_spot_end is None or t_spot_end == 0.:
        t_spot_end = eps
    
    def weight_scheduling_fn(w, t):
        return w * 0.1 ** t

    def conditional_predictor_update_fn(score_model, classifier, x, t, labels, maskX, classifier_scale, *args, **kwargs):
        """The predictor update function for class-conditional sampling."""
        score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous=continuous)
        # The gradient function of the noise-dependent classifier
        classifier_grad_fn = mutils.get_classifier_grad_fn(sde, classifier, train=False, continuous=continuous, 
                                                           regress=regress, labels=labels)

        def total_grad_fn(x, t, *args, **kwargs):
            
            # score =  score_fn(x, t, *args, **kwargs)
            score =  score_fn(x, t, maskX)
            classifier_grad = classifier_grad_fn(x, t, maskX, *args, **kwargs)
            
            # Sample weight
            if weight_scheduling:
                w = weight_scheduling_fn(classifier_scale, t[0].item())
            else:
                w = classifier_scale
            
            if weight_ratio:
                if prod_w:
                    ratio = score.view(x.shape[0], -1).norm(p=p, dim=-1) / (w * classifier_grad).view(x.shape[0], -1).norm(p=p, dim=-1)
                else:
                    ratio = score.view(x.shape[0], -1).norm(p=p, dim=-1) / classifier_grad.view(x.shape[0], -1).norm(p=p, dim=-1)
                # ratio = score.view(x.shape[0], -1).norm(p=p, dim=-1) / classifier_grad.view(x.shape[0], -1).norm(p=p, dim=-1)
                w *= ratio[:, None, None]
            
            if weight_ratio_abs:
                assert not weight_ratio
                ratio = torch.div(torch.abs(score), torch.abs(classifier_grad))
                w *= ratio
            
            score_grad_norm_p.append(torch.mean(score.view(x.shape[0], -1).norm(p=p, dim=-1)).item())
            
            if weight_ratio: # ratio per sample
                classifier_grad_norm_p.append(torch.mean(classifier_grad.view(x.shape[0], -1).norm(p=p, dim=-1) * ratio[:, None, None]).item())
            elif weight_ratio_abs: # ratio per element
                classifier_grad_norm_p.append(torch.mean((classifier_grad * ratio).norm(p=p)).item())
            else:
                classifier_grad_norm_p.append(torch.mean(classifier_grad.view(x.shape[0], -1).norm(p=p, dim=-1)).item())

            if t_spot < 1.:
                if t[0].item() <= t_spot and t[0] >= t_spot_end:
                    return score + w * classifier_grad
                    # return (1 - w) * score + w * classifier_grad
                else:
                    return score
            else:
                # return (1 - w) * score + w * classifier_grad
                return score + w * classifier_grad

        if predictor is None:
            predictor_obj = NonePredictor(sde, total_grad_fn, probability_flow)
        else:
            predictor_obj = predictor(sde, total_grad_fn, probability_flow)
        return predictor_obj.update_fn(x, t, *args, **kwargs)

    def conditional_corrector_update_fn(score_model, classifier, x, t, labels, maskX, classifier_scale, *args, **kwargs):
        """The corrector update function for class-conditional sampling."""
        score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous=continuous)
        classifier_grad_fn = mutils.get_classifier_grad_fn(sde, classifier, train=False, continuous=continuous, 
                                                           regress=regress, labels=labels)

        def total_grad_fn(x, t, *args, **kwargs):
            # score =  score_fn(x, t, *args, **kwargs)
            score =  score_fn(x, t, maskX)
            classifier_grad = classifier_grad_fn(x, t, maskX, *args, **kwargs)
            
            # Sample weight
            if weight_scheduling:
                w = weight_scheduling_fn(classifier_scale, t[0].item())
            else:
                w = classifier_scale
            
            if weight_ratio:
                if prod_w:
                    ratio = score.view(x.shape[0], -1).norm(p=p, dim=-1) / (w * classifier_grad).view(x.shape[0], -1).norm(p=p, dim=-1)
                else:
                    ratio = score.view(x.shape[0], -1).norm(p=p, dim=-1) / classifier_grad.view(x.shape[0], -1).norm(p=p, dim=-1)
                # ratio = score.view(x.shape[0], -1).norm(p=p, dim=-1) / classifier_grad.view(x.shape[0], -1).norm(p=p, dim=-1)
                w *= ratio[:, None, None]

            score_grad_norm_c.append(torch.mean(score.view(x.shape[0], -1).norm(p=p, dim=-1)).item())
            
            if weight_ratio:
                classifier_grad_norm_c.append(torch.mean(classifier_grad.view(x.shape[0], -1).norm(p=p, dim=-1) * ratio[:, None, None]).item())
            else:
                classifier_grad_norm_c.append(torch.mean(classifier_grad.view(x.shape[0], -1).norm(p=p, dim=-1)).item())
            
            if t_spot < 1.:
                if t[0].item() <= t_spot and t[0] >= t_spot_end:
                    return score + w * classifier_grad
                    # return (1 - w) * score + w * classifier_grad
                else:
                    return score
            else:
                return score + w * classifier_grad
                # return (1 - w) * score + w * classifier_grad
        if corrector is None:
            corrector_obj = NoneCorrector(sde, total_grad_fn, snr, n_steps)
        else:
            corrector_obj = corrector(sde, total_grad_fn, snr, n_steps)
        return corrector_obj.update_fn(x, t, *args, **kwargs)

    def pc_conditional_sampler(score_model, mask, classifier,
                               eval_chain=False, keep_chain=None, 
                               number_chain_steps=None, classifier_scale=None,
                               task=None, sample_bs=None):
        """Generate class-conditional samples with Predictor-Corrector (PC) samplers.

        Args:
          score_model: A `torch.nn.Module` object that represents the training state
            of the score-based model.
          labels: A JAX array of integers that represent the target label of each sample.

        Returns:
          Class-conditional samples.
        """
        
        chain_x = None
        if eval_chain:
            if number_chain_steps is None:
                number_chain_steps = sde.N
            if keep_chain is None:
                keep_chain = shape[0] # all sample
            assert number_chain_steps <= sde.N
            chain_x_size = torch.Size((number_chain_steps, keep_chain, *shape[1:]))
            chain_x = torch.zeros(chain_x_size)
        
        with torch.no_grad():
            
            # ----------- Meta-NAS (START) ---------- #
            # different task embedding in a batch
            # task_batch = []
            # for _ in range(shape[0]):
            #     task_batch.append(test_dataset[0])
            # task = torch.stack(task_batch, dim=0)
            
            if task is None:
                # same task embedding in a batch
                task = test_dataset[0]
                task = task.repeat(shape[0], 1, 1)
                task = task.to(device)
            else:
                task = task.repeat(shape[0], 1, 1)
                task = task.to(device)
                # print(f'Sampling stage')
                # import pdb; pdb.set_trace()
            
            # for accerlerating sampling
            classifier.sample_state = True
            classifier.D_mu = None
            # ----------- Meta-NAS (END) ---------- #
            # import pdb; pdb.set_trace()
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
            
            if len(mask.shape) == 3:
                mask = mask[0]
            mask = mask.unsqueeze(0).repeat(x.size(0), 1, 1) # adj
            
            for i in trange(sde.N, desc='[PC conditional sampling]', position=1, leave=False):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                
                x, x_mean = conditional_corrector_update_fn(score_model, classifier, x, vec_t, labels=labels, maskX=mask, task=task, classifier_scale=classifier_scale)
                x, x_mean = conditional_predictor_update_fn(score_model, classifier, x, vec_t, labels=labels, maskX=mask, task=task, classifier_scale=classifier_scale)
                
                if eval_chain:
                # Save the first keep_chain graphs
                    write_index = number_chain_steps - 1 - int((i * number_chain_steps) // sde.N)
                    # write_index = int((t * number_chain_steps) // sde.T)
                    chain_x[write_index] = inverse_scaler(x_mean if denoise else x)[:keep_chain]
            
            classifier.sample_state = False
            return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1), chain_x, (score_grad_norm_p, classifier_grad_norm_p, score_grad_norm_c, classifier_grad_norm_c)
            
    return pc_conditional_sampler



def get_ode_sampler(sde, shape, inverse_scaler, denoise=False,
                    rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3, device='cuda'):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        shape: A sequence of integers. The expected shape of a single sample.
        inverse_scaler: The inverse data normalizer.
        denoise: If `True`, add one-step denoising to final samples.
        rtol: A `float` number. The relative tolerance level of the ODE solver.
        atol: A `float` number. The absolute tolerance level of the ODE solver.
        method: A `str`. The algorithm used for the black-box ODE solver.
            See the documentation of `scipy.integrate.solve_ivp`.
        eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
        device: PyTorch device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """

    def denoise_update_fn(model, x, mask):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps, mask=mask)
        return x

    def drift_fn(model, x, t, mask):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t, mask=mask)[0]

    def ode_sampler(model, n_nodes_pmf, z=None):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
            model: A score model.
            n_nodes_pmf: Probability mass function of graph nodes.
            z: If present, generate samples from latent code `z`.
        Returns:
            samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distribution of the SDE.
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z

            # Sample the number of nodes
            n_nodes = torch.multinomial(n_nodes_pmf, shape[0], replacement=True)
            mask = torch.zeros((shape[0], shape[-1]), device=device)
            for i in range(shape[0]):
                mask[i][:n_nodes[i]] = 1.
            mask = (mask[:, None, :] * mask[:, :, None]).unsqueeze(1)

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, x, vec_t, mask)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                           rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(model, x, mask)

            x = inverse_scaler(x) * mask
            return x, nfe, n_nodes

    return ode_sampler


def get_diffeq_sampler(sde, shape, inverse_scaler, denoise=False,
                       rtol=1e-5, atol=1e-5, step_size=0.01, method='dopri5', eps=1e-3, device='cuda'):
    """
    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        shape: A sequence of integers. The expected shape of a single sample.
        inverse_scaler: The inverse data normalizer.
        denoise: If `True`, add one-step denoising to final samples.
        rtol: A `float` number. The relative tolerance level of the ODE solver.
        atol: A `float` number. The absolute tolerance level of the ODE solver.
        method: A `str`. The algorithm used for the black-box ODE solver in torchdiffeq.
            See the documentation of `torchdiffeq`. eg: adaptive solver('dopri5', 'bosh3', 'fehlberg2')
        eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
        device: PyTorch device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """

    def denoise_update_fn(model, x, mask):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps, mask=mask)
        return x

    def drift_fn(model, x, t, mask):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t, mask=mask)[0]

    def diffeq_sampler(model, n_nodes_pmf, z=None):
        """The probability flow ODE sampler with ODE solver from torchdiffeq.

        Args:
            model: A score model.
            n_nodes_pmf: Probability mass function of graph nodes.
            z: If present, generate samples from latent code `z`.
        Returns:
            samples, number of function evaluations.
        """
        with torch.no_grad():
            # initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distribution of the SDE.
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z

            # Sample the number of nodes
            n_nodes = torch.multinomial(n_nodes_pmf, shape[0], replacement=True)
            mask = torch.zeros((shape[0], shape[-1]), device=device)
            for i in range(shape[0]):
                mask[i][:n_nodes[i]] = 1.
            mask = (mask[:, None, :] * mask[:, :, None]).unsqueeze(1)

            class ODEfunc(torch.nn.Module):
                def __init__(self):
                    super(ODEfunc, self).__init__()
                    self.nfe = 0

                def forward(self, t, x):
                    self.nfe += 1
                    x = x.reshape(shape)
                    vec_t = torch.ones(shape[0], device=x.device) * t
                    drift = drift_fn(model, x, vec_t, mask)
                    return drift.reshape((-1,))

            # Black-box ODE solver for the probability flow ODE
            ode_func = ODEfunc()
            if method in ['dopri5', 'bosh3', 'fehlberg2']:
                solution = odeint(ode_func, x.reshape((-1,)), torch.tensor([sde.T, eps], device=x.device),
                                  rtol=rtol, atol=atol, method=method,
                                  options={'step_t': torch.tensor([1e-3], device=x.device)})
            elif method in ['euler', 'midpoint', 'rk4', 'explicit_adams', 'implicit_adams']:
                solution = odeint(ode_func, x.reshape((-1,)), torch.tensor([sde.T, eps], device=x.device),
                                  rtol=rtol, atol=atol, method=method,
                                  options={'step_size': step_size})

            x = solution[-1, :].reshape(shape)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(model, x, mask)

            x = inverse_scaler(x) * mask
            return x, ode_func.nfe, n_nodes

    return diffeq_sampler
