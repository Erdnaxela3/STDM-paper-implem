import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from stdm.model.impute import stdm_impute
from stdm.model.training import compute_alpha_t
from stdm.nn import STDM
from stdm.utils.data import OSAPDataLoader


def evaluate_stdm(
    model: STDM,
    dataloader: OSAPDataLoader,
    t_diff: int = 50,
    n_steps: int = 5,
    beta_1: float = 1e-4,
    beta_t_diff: float = 0.5,
    save_original_data: bool = False,
) -> dict:
    """
    Evaluate STDM model on given dataloader.

    Parameters
    ----------
        model: STDM
        dataloader: DataLoader
        t_diff: int, default 50
            maximum diffusion step
        n_steps: int, default 5
            number of steps to take during DDIM process
        beta_1: float, default 1e-4
        beta_t_diff: float, default 0.5

    Returns
    -------
        losses: dict
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    total_error = 0
    total_squared_error = 0
    n_total = 0
    with torch.no_grad():
        for i, (x_0, emb_time, m_miss, m) in enumerate(tqdm(dataloader, desc="Evaluating")):
            x_0, emb_time, m_miss, m = x_0.to(device), emb_time.to(device), m_miss.to(device), m.to(device)
            artificially_masked = m_miss - m

            noise = torch.randn_like(x_0).to(device)
            t = torch.full((x_0.size(0), 1, 1), t_diff, device=device).float()

            alpha_t = compute_alpha_t(t, t_diff=t_diff, beta_1=beta_1, beta_t_diff=beta_t_diff)
            x_t_mask = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
            x_t = m * x_0 + (1 - m) * x_t_mask

            # save for analysis purposes
            if save_original_data:
                if not os.path.exists("analysis"):
                    os.makedirs("analysis")

                assert x_0.shape == m_miss.shape == m.shape

                torch.save(x_0, f"analysis/original_{i}.pt")
                torch.save(artificially_masked, f"analysis/artificial_mask_{i}.pt")
                torch.save(x_t, f"analysis/noised_{i}.pt")

            predicted_x_0 = stdm_impute(
                model=model,
                x_t=x_t,
                emb_time=emb_time,
                m=m,
                t_diff=t_diff,
                n_steps=n_steps,
                beta_1=beta_1,
                beta_t_diff=beta_t_diff,
                save_index=i if save_original_data else None,
            )

            assert x_0.shape == predicted_x_0.shape

            y, y_hat = x_0 * artificially_masked, predicted_x_0 * artificially_masked
            error = y - y_hat

            total_error += error.abs().sum().item()
            total_squared_error += (error**2).sum().item()
            n_total += artificially_masked.sum().item()

    losses = {
        "mse": total_squared_error / n_total,
        "mae": total_error / n_total,
    }
    return losses


def evaluate_mean_imputer(
    dataloader: OSAPDataLoader,
) -> dict:
    """
    Evaluate mean imputer model on given dataloader.

    Parameters
    ----------
        dataloader: DataLoader

    Returns
    -------
        losses: dict
    """
    total_error = 0
    total_squared_error = 0
    n_total = 0
    with torch.no_grad():
        for i, (x_0, emb_time, m_miss, m) in enumerate(tqdm(dataloader, desc="Evaluating")):
            artificially_masked = m_miss - m
            x_masked = x_0 * m
            y_hat = x_masked.mean(dim=1, keepdim=True).expand(-1, -1, x_0.size(2))
            y, y_hat = x_0 * artificially_masked, y_hat * artificially_masked
            error = y - y_hat

            total_error += error.abs().sum().item()
            total_squared_error += (error**2).sum().item()
            n_total += artificially_masked.sum().item()

    losses = {
        "mse": total_squared_error / n_total,
        "mae": total_error / n_total,
    }
    return losses
