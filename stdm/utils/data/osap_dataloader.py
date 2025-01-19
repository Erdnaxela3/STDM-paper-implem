from torch.utils.data import DataLoader
from stdm.utils.data import OSAPDataset
import torch


def mask_batch(m_miss: torch.Tensor, masking_ratio: float = 0.1) -> torch.Tensor:
    """
    Mask a batch of data with a given masking ratio.

    Args:
        m_miss: torch.Tensor
            missing mask.
        masking_ratio: float, default 0.1
            masking ratio.

    Returns:
        Mask
    """
    artificially_masked = 1 - (torch.rand(m_miss.size()) < masking_ratio).int()
    m = torch.logical_and(m_miss, artificially_masked).int()

    return m


class OSAPDataLoader(DataLoader):
    """
    DataLoader for the OSAP model.

    No shuffling is performed, as the data is time-ordered.
    Batching is performed by time periods, non-overlapping.
    """

    def __init__(self, dataset: OSAPDataset, batch_size: int, shuffle_masks: bool = False, masking_ratio: float = 0.1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_masks = shuffle_masks
        self.dataset_length = len(dataset)
        self.masking_ratio = masking_ratio
        self.masks = [None for _ in range(self.dataset_length)]

    def __iter__(self):
        for idx in range(0, len(self.dataset), self.batch_size):
            left_idx, right_idx = idx, min(idx + self.batch_size - 1, self.dataset_length - 1)

            x, time, m_miss = self.dataset.__getitem__([left_idx, right_idx], is_indices=True)

            if self.masks[left_idx] is None or self.shuffle_masks:
                self.masks[left_idx] = mask_batch(m_miss, masking_ratio=self.masking_ratio)

            yield x, time, m_miss, self.masks[left_idx]

    def __len__(self):
        return (self.dataset_length + self.batch_size - 1) // self.batch_size
