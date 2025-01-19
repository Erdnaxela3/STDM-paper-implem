import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from stdm import nn
from stdm.nn import STDMDiffusionLoss


def compute_beta_t(t: torch.Tensor, t_diff: int = 50, beta_1: float = 1e-4, beta_t_diff: float = 0.5) -> torch.Tensor:
    """
    Compute beta_t parameter for diffusion forward process as a PyTorch tensor.

    Parameters
    ----------
        t: torch.Tensor
            [t_1, ..., t_N] not necessarily in [1,2,3,...,t_diff]
        t_diff: int, default 50
            maximum diffusion step
        beta_1: float, default 1e-4
        beta_t_diff: float, default 0.5
    Returns
    -------
        beta_t: torch.Tensor
            [beta_t_1, ..., beta_t_N]
    """
    beta_1s = torch.full_like(t.float(), beta_1)
    beta_t_diffs = torch.full_like(t.float(), beta_t_diff)
    beta_t_sqrt = ((t_diff - t) / (t_diff - 1)) * torch.sqrt(beta_1s) + ((t - 1) / (t_diff - 1)) * torch.sqrt(
        beta_t_diffs
    )
    beta_t = beta_t_sqrt**2
    return beta_t


def compute_alpha_t(t: torch.Tensor, t_diff: int = 50, beta_1: float = 1e-4, beta_t_diff: float = 0.5) -> torch.Tensor:
    """
    Compute alpha_t parameter for diffusion forward process as a PyTorch tensor.
    alpha_t = prod(1 - beta_t) for t in 1, ..., t

    Parameters
    ----------
        t: torch.Tensor
            [t_1, ..., t_N] not necessarily in [1,2,3,...,t_diff]
        t_diff: float, default 50
            maximum diffusion step
        beta_1: float, default 1e-4
        beta_t_diff: float, default 0.5

    Returns
    -------
        alpha_t: torch.Tensor
            [alpha_t_1, ..., alpha_t_N]
    """
    range_t = torch.arange(1, t_diff + 1, device=t.device)
    beta_ts = compute_beta_t(range_t, t_diff, beta_1, beta_t_diff)
    alpha_ts = torch.cumprod(1 - beta_ts, dim=0)

    ret = [alpha_ts[t_i - 1] if t_i > 0 else torch.tensor(1.0).to(t.device) for t_i in t[:, 0, 0].long()]  # shape: (N,)
    # shape: (N, 1, 1)
    return torch.stack(ret).unsqueeze(1).unsqueeze(2)


def train_stdm(
    model: nn.STDM,
    train_loder: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    epoch_print: int = 1,
    epoch_early_stop: int = 40,
    save_model: bool = True,
    save_model_dir: str = None,
    t_diff: int = 50,
    beta_1: float = 1e-4,
    beta_t_diff: float = 0.5,
):
    """
    train all models with a total epoch, with early stop if the performance on the validation set does not
    increase more than a certain number of epochs.
    """
    model.to(device)

    val_loss_not_improved = 0
    best_val_loss = float("inf")
    criterion = STDMDiffusionLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, (x_0, emb_time, m_miss, m) in enumerate(tqdm(train_loder, desc="Train set")):
            x_0, emb_time, m_miss, m = x_0.to(device), emb_time.to(device), m_miss.to(device), m.to(device)
            noise = torch.randn_like(x_0).to(device)
            t = torch.randint(0, t_diff, (x_0.size(0), 1, 1), device=device).float()

            logging.debug(f"original missing records: {(1 - m_miss).sum()}/{m_miss.numel()}")
            logging.debug(f"masked records: {(1 - m).sum()}/{m.numel()}")
            logging.debug(f"artificially masked records: {(1 - m).sum() - (1 - m_miss).sum()}/{m.numel()}")

            alpha_t = compute_alpha_t(t, t_diff, beta_1, beta_t_diff)
            x_t_mask = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
            x_t = m * x_0 + (1 - m) * x_t_mask

            optimizer.zero_grad()
            predicted_noise = model(x_t, t, emb_time, m)
            loss = criterion(m, noise, predicted_noise)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        train_loss /= len(train_loder)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (x_0, emb_time, m_miss, m) in enumerate(tqdm(val_loader, desc="Val set")):
                x_0, emb_time, m_miss, m = x_0.to(device), emb_time.to(device), m_miss.to(device), m.to(device)
                noise = torch.randn_like(x_0).to(device)
                t = torch.randint(0, t_diff, (x_0.size(0), 1, 1), device=device).float()

                alpha_t = compute_alpha_t(t, t_diff, beta_1, beta_t_diff)
                x_t_mask = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
                x_t = m * x_0 + (1 - m) * x_t_mask

                predicted_noise = model(x_t, t, emb_time, m)
                loss = criterion(m, noise, predicted_noise)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if epoch % epoch_print == 0:
            logging.info(f"Epoch {epoch} train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            val_loss_not_improved = 0
            # save model
            if save_model and save_model_dir is not None:
                model_path = f"{save_model_dir}/stdm_epoch_{epoch}_val_loss_{val_loss:.4f}.pt"
                torch.save(model.state_dict(), model_path)
                logging.info(f"Model saved at {model_path}")
        else:
            val_loss_not_improved += 1
            if val_loss_not_improved > epoch_early_stop:
                logging.info("Early stopping")
                break

    return model
