import torch

from stdm.model.training import compute_alpha_t
from stdm.nn import STDM

from stdm.model import IMPUTE_LOG_FILENAME


def compute_x0_t_hat(
    x_t: torch.Tensor,
    t: torch.Tensor,
    predicted_noise: torch.Tensor,
    t_diff: int = 50,
    beta_1: float = 1e-4,
    beta_t_diff: float = 0.5,
) -> torch.Tensor:
    """
    Compute x_0_t_hat: estimation of x_0 given x_t, t and predicted noise used in DDIM process.
    Parameters
    ----------
        x_t: torch.Tensor
            Noised tensor at diffusion step t.
        t: torch.Tensor
            Diffusion step tensor.
        predicted_noise: torch.Tensor
            Predicted noise tensor.
        t_diff: int, default 50
            Maximum diffusion step.
        beta_1: float, default 1e-4
        beta_t_diff: float, default 0.5

    Returns
    -------
        x0_t_hat: torch.Tensor
    """
    alpha_ts = compute_alpha_t(t, t_diff=t_diff, beta_1=beta_1, beta_t_diff=beta_t_diff)
    x0_t_hat = (x_t - torch.sqrt(alpha_ts) * predicted_noise) / torch.sqrt(alpha_ts)
    return x0_t_hat


def ddim_step(
    x_t: torch.Tensor,
    t: torch.Tensor,
    predicted_noise: torch.Tensor,
    s: int,
    t_diff: int = 50,
    beta_1: float = 1e-4,
    beta_t_diff: float = 0.5,
) -> torch.Tensor:
    """
    Compute x_s: of x_s given x_t, t and predicted noise used in DDIM process.
    Parameters
    ----------
        x_t: torch.Tensor
            Noised tensor at diffusion step t.
        t: torch.Tensor
            Diffusion step tensor.
        predicted_noise: torch.Tensor
            Predicted noise tensor.
        s: int
            Step to achieve from t to s.
        t_diff: int, default 50
            Maximum diffusion step.
        beta_1: float, default 1e-4
        beta_t_diff: float, default 0.5

    Returns
    -------
        x_s: torch.Tensor
    """
    s = torch.full_like(t, s, device=t.device)
    alpha_s = compute_alpha_t(s, t_diff=t_diff, beta_1=beta_1, beta_t_diff=beta_t_diff)
    x_0_t_hat = compute_x0_t_hat(x_t, t, predicted_noise)

    noise = torch.randn_like(x_t)
    x_s = torch.sqrt(alpha_s) * x_0_t_hat + torch.sqrt(1 - alpha_s) * noise

    return x_s


def stdm_impute(
    model: STDM,
    x_t: torch.Tensor,
    emb_time: torch.Tensor,
    m: torch.Tensor,
    n_steps: int,
    t_diff: int = 50,
    beta_1: float = 1e-4,
    beta_t_diff: float = 0.5,
    save_index: int | None = None,
) -> torch.Tensor:
    """
    Impute missing values in the input tensor x_t using the STDM model.

    Parameters
    ----------
        model: STDM model.
        x_t: torch.Tensor, Input tensor.
        emb_time: Time embedding tensor.
        m: torch.Tensor, Mask tensor (1 if observed, 0 if missing).
        n_steps: int Size of the subsequence steps.
        t_diff: int, default 50
            Maximum diffusion step.
        beta_1: float, default 1e-4
        beta_t_diff: float, default 0.5

    Returns
    -------
        Imputed tensor with same shape as x_t.
    """
    steps = torch.linspace(0, t_diff - 1, n_steps)
    steps = steps.flip(0).round().long()

    model.eval()
    with torch.no_grad():
        x_t_copy = x_t.clone()
        for s in steps:
            t = torch.ones((x_t_copy.size(0), 1, 1), device=x_t.device) * s
            predicted_noise = model(x_t_copy, t, emb_time, m)
            x_t_copy = ddim_step(x_t_copy, t, predicted_noise, s, t_diff=t_diff, beta_1=beta_1, beta_t_diff=beta_t_diff)
            x_t_copy = m * x_t_copy + (1 - m) * x_t_copy

            # save imputed tensor for analysis purposes
            if save_index is not None:
                assert x_t.shape == x_t_copy.shape == m.shape
                torch.save(x_t_copy, f"analysis/step_{s}_{save_index}.pt")

    return x_t_copy
