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



def generate_uniform_unit_sphere_projections(dim, num_projection=1000, device="cpu"):
    f"""
    Generate random uniform unit sphere projections matrix
    :param dim: dimension of measures
    :param num_projection: number of projection vectors to generate
    :return: projection matrix \in \mathbb R^(num_projection, dim)
    """
    projection_matrix = torch.randn((num_projection, dim), device=device)
    projection_matrix = projection_matrix / torch.sqrt(torch.sum(projection_matrix ** 2, dim=1, keepdim=True))
    return projection_matrix


def quantile_function(qs, cws, xs):
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
    Compute the true Wasserstein distance in special case: One dimensional space
    X and Y can comprises of many measures which each measure is a column of X and Y.
    Illustration: Can compute W_1(X[:,0], Y_[:,0]), ..., W_1(X[:,d], Y_[:,d]) simultaneously
    :param X: M source samples. Has shape == (A, M, d)
    :param Y: N target samples. Has shape == (B, N, d)
    :param p: Wasserstein-p
    :return:
    """

    assert X.shape[-1] == Y.shape[-1], "source and target must have the same"
    num_projections = X.shape[-1]

    num_supports_source = X.shape[-2]
    num_supports_target = Y.shape[-2]

    num_dist_source = X.shape[0]
    num_dist_target = Y.shape[0]


    if a is None and b is None:

        if num_supports_source == num_supports_target:
            "Special case when One dimensional space and number of supports are equal"
            X_sorted, X_rankings = torch.sort(X, dim=-2) # shape == (A, M, num_projections)
            Y_sorted, Y_rankings = torch.sort(Y, dim=-2) # shape == (B, M, num_projections)
            diff_quantiles = torch.abs(X_sorted.unsqueeze(1) - Y_sorted.unsqueeze(0)) # shape == (A, B, M, num_projections)
            if p == 1:
                return torch.mean(diff_quantiles, dim=-2) # shape == (A, B, num_projections)
            return torch.mean(torch.pow(diff_quantiles, p), dim=-2) # shape == (A, B, num_projections)


        else:
            "When number of supports are not equal"
            X_sorted, X_rankings = torch.sort(X, dim=-2)
            Y_sorted, Y_rankings = torch.sort(Y, dim=-2)

            a_cum_weights = torch.linspace(1.0 / num_supports_source, 1.0, steps=num_supports_source).to(device)
            b_cum_weights = torch.linspace(1.0 / num_supports_target, 1.0, steps=num_supports_target).to(device)

            qs = torch.sort(torch.concat((a_cum_weights, b_cum_weights), 0), dim=0, descending=False)[0]

            X_quantiles = quantile_function(qs, a_cum_weights, X_sorted) # shape == (A, M+N, num_projections)
            Y_quantiles = quantile_function(qs, b_cum_weights, Y_sorted) # shape == (B, M+N, num_projections)
            diff_quantiles = torch.abs(X_quantiles.unsqueeze(1) - Y_quantiles.unsqueeze(0)) # shape == (A, B, M+N, num_projections)

            qs = torch.cat((torch.zeros(1), qs), dim=0)
            diff_qs = qs[1:, ...] - qs[:-1, ...]
            diff_qs[torch.abs(diff_qs) < 1e-4] = 0
            delta = diff_qs.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            if p == 1:
                return torch.sum(delta * diff_quantiles, dim=-2)
            return torch.pow(input=torch.sum(delta * torch.pow(diff_quantiles, p), dim=-2), exponent=1/p)


def Sliced_Wasserstein_Distance(X, Y, num_projection=1000, projection_vectors=None, p=2, device="cpu"):
    """
    Compute Sliced Wasserstein Distance in the conventional way. Can back propagate this function
    :param X: a batch of A having M source samples. Has shape == (A, num_supports_source, d)
    :param Y: a batch of B having N target samples. Has shape == (B, num_supports_target, d)
    :param num_projection: number of projection matrix. It is a number
    :param p: Wasserstein-p
    :return: Sliced Wasserstein distance (float)
    """

    assert X.shape[-1] == Y.shape[-1], "source and target must have the same"

    dims = X.shape[-1]

    num_supports_source = X.shape[-2]
    num_supports_target = Y.shape[-2]

    num_dist_source = X.shape[0]
    num_dist_target = Y.shape[0]

    device = X.device

    if projection_vectors is None:
        projection_vectors = generate_uniform_unit_sphere_projections(dim=X.shape[-1],
                                                                      num_projection=num_projection,
                                                                      device=device) # shape == (num_projection, d)

    X_projection = torch.matmul(X, projection_vectors.t().unsqueeze(0))  # shape == (A, num_supports_source, num_projection)
    Y_projection = torch.matmul(Y, projection_vectors.t().unsqueeze(0))  # shape == (B, num_supports_target, num_projection)

#############
    w_1d = Wasserstein_One_Dimension(X=X_projection,
                                     Y=Y_projection,
                                     p=p,
                                     device=device)  # shape (num_projection)
    if p == 1:
        return torch.mean(w_1d)
    sw = torch.pow(input=w_1d, exponent=p)
    return torch.pow(torch.mean(sw, dim=-1), exponent=1/p)


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
    ws = Sliced_Wasserstein_Distance(X=X, Y=Y, num_projection=100000, projection_vectors=projection_vectors, p=p)
    end_time = time.time()

    for i in range(2):
        for j in range(3):
            seq_ws = Sequential_Sliced_Wasserstein_Distance(X=X[i, :], Y=Y[j, :], num_projection=100000, projection_vectors=projection_vectors, p=p)
            print(ws[i, j] - seq_ws)