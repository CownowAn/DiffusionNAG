"""Various sampling methods."""

import functools
import torch
import numpy as np
import abc
from tqdm import trange
import sde_lib
from models import utils as mutils
from datasets_nas import MetaTestDataset
from all_path import DATA_PATH


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
    config, 
    sde, 
    shape, 
    inverse_scaler, 
    eps, 
    conditional=False,
    data_name='cifar10', 
    num_sample=20):
    """Create a sampling function.

    Args:
        config: A `ml_collections.ConfigDict` object that contains all configuration information.
        sde: A `sde_lib.SDE` object that represents the forward SDE.
        shape: A sequence of integers representing the expected shape of a single sample.
        inverse_scaler: The inverse data normalizer function.
        eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.
        conditional: If `True`, the sampling function is conditional
        data_name: A `str` name of the dataset.
        num_sample: An `int` number of samples for each class of the dataset.

    Returns:
        A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """

    sampler_name = config.sampling.method

    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    if sampler_name.lower() == 'pc':
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())

        if not conditional:
            print('>>> Get pc_sampler...')
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
            print('>>> Get pc_conditional_sampler...')
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
                                                        data_name=data_name,
                                                        num_sample=num_sample)

    else:
        raise NotImplementedError(f"Sampler name {sampler_name} unknown.")

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
    
    def update_fn(self, x, t, *args, **kwargs):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t, *args, **kwargs)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
        return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
    
    def update_fn(self, x, t, *args, **kwargs):
        f, G = self.rsde.discretize(x, t, *args, **kwargs)
        z = torch.randn_like(x)
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


def get_pc_sampler(sde, 
                   shape, 
                   predictor, 
                   corrector, 
                   inverse_scaler, 
                   snr,
                   n_steps=1, 
                   probability_flow=False, 
                   continuous=False,
                   denoise=True, 
                   eps=1e-3, 
                   device='cuda'):
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


def get_pc_sampler_nas(sde, 
                       shape, 
                       predictor, 
                       corrector, 
                       inverse_scaler, 
                       snr,
                       n_steps=1, 
                       probability_flow=False, 
                       continuous=False,
                       denoise=True, 
                       eps=1e-3, 
                       device='cuda'):
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
            mask = mask[0].unsqueeze(0).repeat(x.size(0), 1, 1)

            for i in trange(sde.N, desc='[PC sampling]', position=1, leave=False):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = corrector_update_fn(x, vec_t, model=model, maskX=mask)
                x, x_mean = predictor_update_fn(x, vec_t, model=model, maskX=mask)
            return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1), None

    return pc_sampler


def get_pc_conditional_sampler_meta_nas(
    sde, 
    shape,
    predictor,
    corrector, 
    inverse_scaler, 
    snr,
    n_steps=1, 
    probability_flow=False,
    continuous=False, 
    denoise=True, 
    eps=1e-5, 
    device='cuda',
    regress=True, 
    labels='max',
    data_name='cifar10', 
    num_sample=20):

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

    # --------- Meta-NAS ---------- #
    test_dataset = MetaTestDataset(
        data_path=DATA_PATH,
        data_name=data_name,
        num_sample=num_sample)


    def conditional_predictor_update_fn(score_model, classifier, x, t, labels, maskX, classifier_scale, *args, **kwargs):
        """The predictor update function for class-conditional sampling."""
        score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous=continuous)
        classifier_grad_fn = mutils.get_classifier_grad_fn(sde, classifier, train=False, continuous=continuous, 
                                                           regress=regress, labels=labels)

        def total_grad_fn(x, t, *args, **kwargs):
            score =  score_fn(x, t, maskX)
            classifier_grad = classifier_grad_fn(x, t, maskX, *args, **kwargs)
            return score + classifier_scale * classifier_grad

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
            score =  score_fn(x, t, maskX)
            classifier_grad = classifier_grad_fn(x, t, maskX, *args, **kwargs)
            return score + classifier_scale * classifier_grad

        if corrector is None:
            corrector_obj = NoneCorrector(sde, total_grad_fn, snr, n_steps)
        else:
            corrector_obj = corrector(sde, total_grad_fn, snr, n_steps)

        return corrector_obj.update_fn(x, t, *args, **kwargs)


    def pc_conditional_sampler(
        score_model, 
        mask,
        classifier,
        classifier_scale=None,
        task=None):

        """Generate class-conditional samples with Predictor-Corrector (PC) samplers.

        Args:
          score_model: A `torch.nn.Module` object that represents the training state
            of the score-based model.
          labels: A JAX array of integers that represent the target label of each sample.

        Returns:
          Class-conditional samples.
        """

        # to accerlerating sampling
        with torch.no_grad():
            if task is None:
                task = test_dataset[0]
                task = task.repeat(shape[0], 1, 1)
                task = task.to(device)
            else:
                task = task.repeat(shape[0], 1, 1)
                task = task.to(device)
            classifier.sample_state = True
            classifier.D_mu = None

            # initial sample
            x = sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            if len(mask.shape) == 3: mask = mask[0]
            mask = mask.unsqueeze(0).repeat(x.size(0), 1, 1) # adj

            for i in trange(sde.N, desc='[PC conditional sampling]', position=1, leave=False):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = conditional_corrector_update_fn(score_model, classifier, x, vec_t, labels=labels, maskX=mask, task=task, classifier_scale=classifier_scale)
                x, x_mean = conditional_predictor_update_fn(score_model, classifier, x, vec_t, labels=labels, maskX=mask, task=task, classifier_scale=classifier_scale)
            classifier.sample_state = False
            return inverse_scaler(x_mean if denoise else x)

    return pc_conditional_sampler