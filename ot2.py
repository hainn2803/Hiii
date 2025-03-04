import torch
import ot
import time
import numpy


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
    n = xs.shape[0]
    cws = cws.t().contiguous()
    qs = qs.t().contiguous()
    idx = torch.searchsorted(cws, qs).t()
    return torch.take_along_dim(input=xs, indices=idx, dim=0)



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
    :param X: M source samples. Has shape == (M, d)
    :param Y: N target samples. Has shape == (N, d)
    :param p: Wasserstein-p
    :return:
    """

    num_supports_source = X.shape[0]
    num_supports_target = Y.shape[0]

    if X.ndim == 1:
        X = X.unsqueeze(1)
    if Y.ndim == 1:
        Y = Y.unsqueeze(1)

    assert X.shape[1] == Y.shape[1], "X and Y must have same number of measures"

    if num_supports_source == num_supports_target and a is None and b is None:
        "Special case when One dimensional space and number of supports are equal"
        X_sorted, X_rankings = torch.sort(X, dim=0)
        Y_sorted, Y_rankings = torch.sort(Y, dim=0)
        diff_quantiles = torch.abs(X_sorted - Y_sorted)
        if p == 1:
            return torch.mean(diff_quantiles, dim=0)
        return torch.mean(torch.pow(diff_quantiles, p), dim=0)

    else:
        "When number of supports are not equal"
        if a is None:
            a = torch.full(X.shape, 1.0 / num_supports_source, device=device)
        elif a.ndim != X.ndim:
            a = a.unsqueeze(1).expand(-1, X.shape[1])
        if b is None:
            b = torch.full(Y.shape, 1.0 / num_supports_target, device=device)
        elif b.ndim != Y.ndim:
            b = b.unsqueeze(1).expand(-1, Y.shape[1])

        X_sorted, X_rankings = torch.sort(X, dim=0)
        Y_sorted, Y_rankings = torch.sort(Y, dim=0)

        a = torch.take_along_dim(input=a, indices=X_rankings, dim=0)  # reorder weight measure corresponding to X_sorted
        b = torch.take_along_dim(input=b, indices=Y_rankings, dim=0)  # reorder weight measure corresponding to Y_sorted

        a_cum_weights = torch.cumsum(a, dim=0)
        b_cum_weights = torch.cumsum(b, dim=0)

        qs = torch.sort(torch.concat((a_cum_weights, b_cum_weights), 0), dim=0, descending=False)[0]
        # qs has shape (num_supports_source + num_supports_target, d)
        # print(qs[:,0])

        # torch.quantile(input=X_sorted, q=qs, dim=0, keepdim=False, interpolation='linear')
        X_quantiles = quantile_function(qs, a_cum_weights, X_sorted)
        Y_quantiles = quantile_function(qs, b_cum_weights, Y_sorted)
        diff_quantiles = torch.abs(X_quantiles - Y_quantiles)

        zeros = torch.zeros((1, qs.shape[1]))
        qs = torch.cat((zeros, qs), dim=0)

        delta = qs[1:, ...] - qs[:-1, ...]
        if p == 1:
            return torch.sum(delta * diff_quantiles, dim=0)
        return torch.pow(input=torch.sum(delta * torch.pow(diff_quantiles, p), dim=0), exponent=1/p)


def Sequential_Sliced_Wasserstein_Distance(X, Y, a=None, b=None, num_projection=1000, projection_vectors=None, p=2, device="cpu"):
    """
    Compute Sliced Wasserstein Distance in the conventional way. Can back propagate this function
    :param X: M source samples. Has shape == (num_supports_source, d)
    :param Y: N target samples. Has shape == (num_supports_target, d)
    :param num_projection: number of projection matrix. It is a number
    :param p: Wasserstein-p
    :return: Sliced Wasserstein distance (float)
    """

    assert X.shape[1] == Y.shape[1], "source and target must have the same"

    num_supports_source = X.shape[0]
    num_supports_target = Y.shape[0]

    if a is None:
        a = torch.full((num_supports_source,), 1.0 / num_supports_source, device=device)  # source measures

    if b is None:
        b = torch.full((num_supports_target,), 1.0 / num_supports_target, device=device)  # target measures

    if projection_vectors is None:
        projection_vectors = generate_uniform_unit_sphere_projections(dim=X.shape[1],
                                                                      num_projection=num_projection,
                                                                      device=device)  # shape == (num_projection, d)

    X_projection = torch.matmul(X, projection_vectors.t())  # shape == (num_supports_source, num_projection)
    Y_projection = torch.matmul(Y, projection_vectors.t())  # shape == (num_supports_target, num_projection)

    w_1d = Wasserstein_One_Dimension(X=X_projection,
                                     Y=Y_projection,
                                     a=a,
                                     b=b,
                                     p=p,
                                     device=device)  # shape (num_projection)

    if p == 1:
        return torch.mean(w_1d)
    sw = torch.pow(input=w_1d, exponent=p)
    return torch.pow(torch.mean(sw), exponent=1/p)


if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)

    M = 5
    N = 4
    d = 50
    p = 2

    X = torch.rand(M, d)
    Y = torch.rand(N, d)
    start_time = time.time()
    ws = Sliced_Wasserstein_Distance(X=X, Y=Y, num_projection=10, p=p)
    end_time = time.time()
    print(f"{ws}, executing time: {end_time - start_time}")