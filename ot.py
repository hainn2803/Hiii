import torch
import torch.nn as nn
import ot
import time
import numpy
from ot2 import Sequential_Sliced_Wasserstein_Distance


class SinkhornAlgorithm(nn.Module):

    def __init__(self, epsilon=0.1, iterations=100, threshold=1e-9):
        super(SinkhornAlgorithm, self).__init__()
        self.epsilon = epsilon
        self.iterations = iterations
        self.threshold = threshold

    def _compute_matrix_H(self, u, v, cost_matrix):
        kernel = -cost_matrix + u.unsqueeze(-1) + v.unsqueeze(-2)
        kernel /= self.epsilon
        return kernel

    def forward(self, p, q, cost_matrix):

        u = torch.zeros_like(p)
        v = torch.zeros_like(q)

        for i in range(self.iterations):
            old_u = u
            old_v = v

            H = self._compute_matrix_H(u, v, cost_matrix)
            u = self.epsilon * (torch.log(p + 1e-8) - torch.logsumexp(H, dim=-1)) + u

            if H.ndim == 3:
                H = self._compute_matrix_H(u, v, cost_matrix).permute(0, 2, 1)
            else:
                H = self._compute_matrix_H(u, v, cost_matrix).permute(1, 0)

            v = self.epsilon * (torch.log(q + 1e-8) - torch.logsumexp(H, dim=-1)) + v

            diff = torch.sum(torch.abs(u - old_u), dim=-1) + torch.sum(torch.abs(v - old_v), dim=-1)
            mean_diff = torch.mean(diff)

            if mean_diff.item() < self.threshold:
                break

        K = self._compute_matrix_H(u, v, cost_matrix)
        pi = torch.exp(K)

        return pi



def generate_uniform_unit_sphere_projections(dim, num_projection=1000, dtype=torch.float32, device="cpu"):
    """
    Generate random uniform unit sphere projections with the same dtype as X and Y.
    """
    projection_matrix = torch.randn((num_projection, dim), dtype=dtype, device=device)
    return projection_matrix / torch.linalg.norm(projection_matrix, dim=1, keepdim=True)


def quantile_function(qs, cws, xs):
    cws, _ = torch.sort(cws, dim=0)
    qs, _ = torch.sort(qs, dim=0)
    num_dist = xs.shape[0]
    num_projections = xs.shape[-1]
    cws = cws.t().contiguous()
    qs = qs.t().contiguous()
    idx = torch.searchsorted(cws, qs).t()
    return torch.take_along_dim(input=xs, indices=idx.expand(num_projections, idx.shape[-1]).t().expand(num_dist, idx.shape[-1], num_projections), dim=-2)


def Wasserstein_Distance(X, Y, p=2, device="cpu"):
    """
    Compute the true Wasserstein distance. Can back propagate this function
    Computational complexity: O(n^3)
    :param X: M source samples. Has shape == (M, d)
    :param Y: N target samples. Has shape == (N, d)
    :param p: Wasserstein-p
    :return: Wasserstein distance (OT cost) == M * T. It is a number
    """

    assert X.shape[1] == Y.shape[1], "source and target must have the same"

    # cost matrix between source and target. Has shape == (M, N)
    M = ot.dist(x1=X, x2=Y, metric='sqeuclidean', p=p, w=None)

    num_supports_source = X.shape[0]
    num_supports_target = Y.shape[0]

    a = torch.full((num_supports_source,), 1.0 / num_supports_source, device=device)
    b = torch.full((num_supports_target,), 1.0 / num_supports_target, device=device)

    ws = ot.emd2(a=a,
                 b=b,
                 M=M,
                 processes=1,
                 numItermax=100000,
                 log=False,
                 return_matrix=False,
                 center_dual=True,
                 numThreads=1,
                 check_marginals=True)

    return ws


def Wasserstein_One_Dimension(X, Y, a=None, b=None, p=2, device="cpu"):
    """
    Compute the true Wasserstein distance in one-dimensional space.
    :param X: Source samples, shape (A, M, d)
    :param Y: Target samples, shape (B, N, d)
    :param p: Wasserstein-p order
    :return: Tensor of shape (A, B, d) with Wasserstein distances
    """
    assert X.shape[-1] == Y.shape[-1], "Source and target must have the same dimension"
    num_projections = X.shape[-1]
    num_supports_source = X.shape[-2]
    num_supports_target = Y.shape[-2]

    num_dist_source = X.shape[0]
    num_dist_target = Y.shape[0]

    if a is None and b is None:
        if num_supports_source == num_supports_target:
            # Equal supports case
            X_sorted, _ = torch.sort(X, dim=-2)  # shape (A, M, d)
            Y_sorted, _ = torch.sort(Y, dim=-2)  # shape (B, N, d)
            diff_quantiles = torch.abs(X_sorted.unsqueeze(1) - Y_sorted.unsqueeze(0))  # shape (A, B, M, d)
            if p == 1:
                return torch.mean(diff_quantiles, dim=-2)  # shape (A, B, d)
            return torch.pow(torch.mean(torch.pow(diff_quantiles, p), dim=-2), 1/p)  # shape (A, B, d)
        else:
            vectorize = True
            # Unequal supports case
            if vectorize is True:
                X_sorted, _ = torch.sort(X, dim=-2)  # shape (A, M, d)
                Y_sorted, _ = torch.sort(Y, dim=-2)  # shape (B, N, d)

                a_cum_weights = torch.linspace(1.0 / num_supports_source, 1.0, steps=num_supports_source).to(device)
                b_cum_weights = torch.linspace(1.0 / num_supports_target, 1.0, steps=num_supports_target).to(device)
                qs = torch.sort(torch.cat((a_cum_weights, b_cum_weights), 0), dim=0, descending=False)[0]

                X_quantiles = quantile_function(qs, a_cum_weights, X_sorted)  # shape (A, len(qs), d)
                Y_quantiles = quantile_function(qs, b_cum_weights, Y_sorted)  # shape (B, len(qs), d)

                # del a_cum_weights
                # del b_cum_weights
                # del X_sorted
                # del Y_sorted

                diff_quantiles = torch.abs(X_quantiles.unsqueeze(1) - Y_quantiles.unsqueeze(0))  # shape (A, B, len(qs), d)

                # del X_quantiles
                # del Y_quantiles

                qs_extended = torch.cat((torch.zeros(1).to(device), qs), dim=0)
                diff_qs = qs_extended[1:] - qs_extended[:-1]
                diff_qs = torch.clamp(diff_qs, min=1e-6)
                delta = diff_qs.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # shape (1, 1, len(qs), 1)

                if p == 1:
                    return torch.sum(delta * diff_quantiles, dim=-2)  # shape (A, B, d)
                else:
                    weighted_sum = torch.sum(delta * torch.pow(diff_quantiles, p), dim=-2)  # shape (A, B, d)
                    return torch.pow(weighted_sum, 1/p)

            else:
                X_sorted, _ = torch.sort(X, dim=-2)
                Y_sorted, _ = torch.sort(Y, dim=-2)
                a_cum_weights = torch.linspace(1.0 / num_supports_source, 1.0, num_supports_source).to(device)
                b_cum_weights = torch.linspace(1.0 / num_supports_target, 1.0, num_supports_target).to(device)
                qs = torch.sort(torch.cat((a_cum_weights, b_cum_weights), 0))[0]
                
                result = torch.zeros(num_dist_source, num_dist_target, num_projections, device=device)

                for i in range(num_dist_source):
                    X_q = quantile_function(qs, a_cum_weights, X_sorted[i].unsqueeze(0))
                    for j in range(num_dist_target):
                        Y_q = quantile_function(qs, b_cum_weights, Y_sorted[j].unsqueeze(0))
                        diff = torch.abs(X_q.squeeze(0) - Y_q.squeeze(0))
                        qs_extended = torch.cat((torch.zeros(1).to(device), qs), dim=0)
                        diff_qs = qs_extended[1:] - qs_extended[:-1]
                        diff_qs[torch.abs(diff_qs) < 1e-6] = 0
                        delta = diff_qs.unsqueeze(-1)
                        if p == 1:
                            result[i, j, :] = torch.sum(delta * diff, dim=0) 
                        else:
                            result[i, j, :] = torch.pow(torch.sum(delta * torch.pow(diff, p), dim=0), 1/p)
                        del Y_q, diff
                    del X_q
                return result
    raise NotImplementedError("Weighted Wasserstein not implemented")


def Sliced_Wasserstein_Distance(X, Y, num_projections=1000, list_projection_vectors=None, p=2, device="cpu", chunk=1000):
    """
    Compute Sliced Wasserstein Distance efficiently. Supports backpropagation.
    
    :param X: Batch of A source measures, shape (A, num_supports_source, d)
    :param Y: Batch of B target measures, shape (B, num_supports_target, d)
    :param num_projections: Number of projection directions
    :param list_projection_vectors: Optional precomputed projection vectors
    :param p: Wasserstein-p parameter (e.g., 1 or 2)
    :param device: Device to perform computation on (e.g., "cpu" or "cuda")
    :param chunk: Number of projections per chunk for memory management
    :return: Tensor of shape (A, B) with Sliced Wasserstein Distances
    """
    assert X.shape[-1] == Y.shape[-1], "Source and target must have the same dimension"

    dims = X.shape[-1]
    num_dist_source = X.shape[0]
    num_dist_target = Y.shape[0]

    # Adjust chunk size if num_projections is smaller
    if num_projections < chunk:
        chunk = num_projections
        chunk_num_projections = 1
    else:
        chunk_num_projections = num_projections // chunk

    # Initialize running sum for p-th powers of 1D Wasserstein distances
    sum_w_p = torch.zeros((num_dist_source, num_dist_target), device=device)

    for i in range(chunk_num_projections):
        # Generate or retrieve projection vectors
        if list_projection_vectors is None:
            projection_vectors = generate_uniform_unit_sphere_projections(
                dim=dims, num_projection=chunk, device=device
            ).detach()

        else:
            projection_vectors = list_projection_vectors[i]

        # Project X and Y onto 1D
        X_projection = torch.einsum('bnd,pd->bnp', X.to(projection_vectors.dtype), projection_vectors)
        Y_projection = torch.einsum('bmd,pd->bmp', Y.to(projection_vectors.dtype), projection_vectors)

        del projection_vectors

        # Compute 1D Wasserstein distance for this chunk
        torch.cuda.empty_cache()
        w_1d = Wasserstein_One_Dimension(
            X=X_projection, Y=Y_projection, p=p, device=device
        ).to(device)  # Shape: (A, B, chunk)

        # Accumulate sum of p-th powers along projection dimension
        sum_w_p += torch.sum(torch.pow(w_1d, p), dim=-1)

        del X_projection, Y_projection, w_1d

    # Compute the final Sliced Wasserstein Distance
    mean_w_p = sum_w_p / num_projections
    if p == 1:
        return mean_w_p  # Shape: (A, B)
    else:
        return torch.pow(mean_w_p, 1/p)  # Shape: (A, B)



if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)

    M = 10
    N = 4
    d = 8
    p = 2

    X = torch.rand(2, M, d)
    Y = torch.rand(3, N, d)

    projection_vectors = generate_uniform_unit_sphere_projections(dim=X.shape[-1],
                                                                    num_projection=100000)

    start_time = time.time()
    ws = Sliced_Wasserstein_Distance(X=X, Y=Y, num_projections=100000, p=p)
    end_time = time.time()

    # for i in range(2):
    #     for j in range(3):
    #         seq_ws = Sequential_Sliced_Wasserstein_Distance(X=X[i, :], Y=Y[j, :], num_projection=100000, projection_vectors=projection_vectors, p=p)
    #         print(ws[i, j] - seq_ws)