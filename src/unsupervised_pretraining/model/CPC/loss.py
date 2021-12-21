from collections import OrderedDict

import numpy as np
import torch


class BilinearComparator(torch.nn.Module):
    def __init__(self, x_dim, y_dim, config=None):
        super().__init__()
        self.transform = torch.nn.Linear(x_dim, y_dim, bias=False)

    def forward(self, x, y):
        """Embeddings should be (batch, time, dim) or (batch, dim)."""
        if len(x.shape) != len(y.shape):
            raise ValueError("Input rank mismatch: {} != {}".format(
                len(x.shape), len(y.shape)))
        has_time = (len(x.shape) == 3)
        if has_time:
            if x.shape[1] != y.shape[1]:
                raise ValueError("Time dimension mismatch: {} != {}".format(
                    x.shape[1], y.shape[1]))
        input_x_shape = x.shape
        input_y_shape = y.shape
        x = self.transform(x).reshape(-1, input_y_shape[-1])
        y = y.reshape(-1, input_y_shape[-1])
        out = torch.matmul(x, y.t())  # (batch, batch) or (batch x time, batch x time).
        if has_time:
            batch_size, duration = input_x_shape[:2]
            out = out.view(batch_size, duration, batch_size, duration)
        return out


class NamedLoss(torch.nn.Module):
    input_names = None


class InfoNCELoss(NamedLoss):
    input_names = ("embeddings", "contexts")

    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("min_context_size", 1),
            ("future_steps", 5),
            ("comparator_params", None)
        ])

    def __init__(self, embeddings_size, contexts_size, config=None):
        super().__init__()
        self._config = self.get_default_config()
        for step in range(self._config["future_steps"]):
            comparator = BilinearComparator(embeddings_size, contexts_size,
                                            config=self._config["comparator_params"])
            self.add_module("comparator{}".format(step), comparator)

    def forward(self, embeddings, contexts):
        if len(embeddings.shape) != 3:
            raise ValueError("Expected embeddings with shape (batch, time, dim")
        if len(contexts.shape) != 3:
            raise ValueError("Expected contexts with shape (batch, time, dim")
        if embeddings.shape[:2] != contexts.shape[:2]:
            raise ValueError("Embeddings and contexts shape mismatch: {} != {}".format(
                embeddings.shape[:2], contexts.shape[:2]))
        batch_size, duration = embeddings.shape[:2]
        min_context_size = self._config["min_context_size"]
        future_steps = self._config["future_steps"]
        if embeddings.shape[1] != contexts.shape[1]:
            raise ValueError("Features and contexts duration mismatch")
        if embeddings.shape[1] < min_context_size + future_steps:
            raise ValueError("Duration is not enough for InfoNCE loss")
        flat_log_probabilities_array = []
        for step in range(future_steps):
            embeddings_subset = embeddings[:, min_context_size + step:duration]
            contexts_subset = contexts[:, min_context_size - 1:duration - step - 1]
            subset_duration = duration - step - min_context_size
            comparator = self._modules["comparator{}".format(step)]
            log_density_ratios_matrix = comparator(embeddings_subset, contexts_subset)  # (batch, time, batch, time).
            # Numerator consists from ratios for matching (batch, time) pairs.
            log_density_ratios_positive = log_density_ratios_matrix.view(batch_size * subset_duration, batch_size * subset_duration).diag().view(batch_size, subset_duration)
            # Negatives are obtained from different batch elements for the same time step.
            # Denominator is just a sum of ratios for different samples from the batch.
            log_density_ratios_alt = torch.diagonal(log_density_ratios_matrix, dim1=1, dim2=3)  # (batch, batch, time).
            log_density_ratio_sums = torch.logsumexp(log_density_ratios_alt, dim=1)  # (batch, time).
            # We implement mean instead of sum for better loss values. It is not part of original approach.
            log_density_ratio_sums = log_density_ratio_sums - np.log(batch_size * subset_duration)
            log_probabilities = log_density_ratios_positive - log_density_ratio_sums  # (batch, time).
            flat_log_probabilities_array.append(log_probabilities.flatten())  # (batch x time).
        flat_log_probabilities = torch.cat(flat_log_probabilities_array, dim=0)
        losses = -flat_log_probabilities
        return losses