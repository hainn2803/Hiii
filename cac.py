import torch

def generate_uniform_unit_sphere_projections(dim, num_projection=1000, device="cpu"):
    """
    Generate random uniform unit sphere projections matrix
    :param dim: dimension of measures
    :param num_projection: number of projection vectors to generate
    :return: projection matrix ∈ ℝ^(num_projection, dim)
    """
    projection_matrix = torch.randn((num_projection, dim), device=device)
    return projection_matrix / torch.linalg.norm(projection_matrix, dim=1, keepdim=True)


def quantile_function(qs, cws, xs):
    cws, _ = torch.sort(cws, dim=0)
    qs, _ = torch.sort(qs, dim=0)
    
    num_dist, num_supports, num_projections = xs.shape
    cws, qs = cws.t().contiguous(), qs.t().contiguous()
    
    idx = torch.searchsorted(cws, qs).t().unsqueeze(-1).expand(-1, -1, num_projections)
    return torch.take_along_dim(xs, indices=idx.expand(num_dist, -1, num_projections), dim=-2)


def Wasserstein_One_Dimension(X, Y, a=None, b=None, p=2, device="cpu"):
    """
    Compute the Wasserstein distance in one-dimensional space.
    :param X: Source samples, shape (A, M, d)
    :param Y: Target samples, shape (B, N, d)
    :param p: Wasserstein-p order
    :return: Tensor of shape (A, B, d) with Wasserstein distances
    """
    assert X.shape[-1] == Y.shape[-1], "Source and target must have the same dimension"
    
    num_dist_source, num_supports_source, num_projections = X.shape
    num_dist_target, num_supports_target, _ = Y.shape

    X_sorted, _ = torch.sort(X, dim=-2)
    Y_sorted, _ = torch.sort(Y, dim=-2)

    if num_supports_source == num_supports_target:
        diff_quantiles = torch.abs(X_sorted.unsqueeze(1) - Y_sorted.unsqueeze(0))
        return torch.pow(torch.mean(torch.pow(diff_quantiles, p), dim=-2), 1/p) if p > 1 else torch.mean(diff_quantiles, dim=-2)

    a_cum_weights = torch.linspace(1.0 / num_supports_source, 1.0, num_supports_source, device=device)
    b_cum_weights = torch.linspace(1.0 / num_supports_target, 1.0, num_supports_target, device=device)
    qs = torch.sort(torch.cat((a_cum_weights, b_cum_weights), dim=0))[0]

    X_quantiles = quantile_function(qs, a_cum_weights, X_sorted)
    Y_quantiles = quantile_function(qs, b_cum_weights, Y_sorted)
    
    diff_quantiles = torch.abs(X_quantiles.unsqueeze(1) - Y_quantiles.unsqueeze(0))

    qs_extended = torch.cat((torch.zeros(1, device=device), qs), dim=0)
    diff_qs = torch.clamp(qs_extended[1:] - qs_extended[:-1], min=1e-6).unsqueeze(0).unsqueeze(0).unsqueeze(-1)

    return torch.pow(torch.sum(diff_qs * torch.pow(diff_quantiles, p), dim=-2), 1/p) if p > 1 else torch.sum(diff_qs * diff_quantiles, dim=-2)


def Sliced_Wasserstein_Distance(X, Y, num_projections=1000, list_projection_vectors=None, p=2, device="cpu", chunk=1000):
    """
    Compute Sliced Wasserstein Distance efficiently. Supports backpropagation through X and Y.
    
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

    chunk = min(chunk, num_projections)
    chunk_num_projections = (num_projections + chunk - 1) // chunk  # Ensure rounding up

    sum_w_p = torch.zeros((num_dist_source, num_dist_target), device=device)

    for _ in range(chunk_num_projections):
        # Generate projection vectors (detached for memory efficiency)
        projection_vectors = generate_uniform_unit_sphere_projections(dims, num_projection=chunk, device=device).detach()
        
        # Project X and Y onto 1D
        X_projection = torch.einsum('bnd,pd->bnp', X, projection_vectors)
        Y_projection = torch.einsum('bmd,pd->bmp', Y, projection_vectors)

        # Compute 1D Wasserstein distance for this chunk
        w_1d = Wasserstein_One_Dimension(X_projection, Y_projection, p=p, device=device)

        sum_w_p += torch.sum(torch.pow(w_1d, p), dim=-1)

    mean_w_p = sum_w_p / num_projections
    return torch.pow(mean_w_p, 1/p) if p > 1 else mean_w_p
