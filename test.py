import torch 


# x = torch.randn(2, 4, 5)
# y = torch.randn(10, 5)
# # want z to be z.shape == (2, 4, 10)

# z = torch.matmul(x, y.t().unsqueeze(0))
# print(z.shape)

# x = torch.randn(10, 4, 5)
# y = torch.randn(15, 4, 5)

# z = x.unsqueeze(1) - y.unsqueeze(0)
# print(z.shape)

# t = z[0, 1, :, :] - (x[0] - y[1])
# print(t)

X = torch.randn(3, 4, 5)

a = torch.randn(2, 6, 5)

# print(X)
# print(a)

# X_sorted, X_rankings = torch.sort(X, dim=-2)
# a = torch.take_along_dim(input=a, indices=X_rankings, dim=0)  # reorder weight measure corresponding to X_sorted

# print(X_rankings)
# print(a)

# t = torch.concat((X.unsqueeze(1).repeat(1, 2, 1, 1), a.unsqueeze(0).repeat(3, 1, 1, 1)), dim=-2)

# print(t.shape)



def quantile_function(qs, cws, xs):
    num_dist = xs.shape[0]
    num_projections = xs.shape[-1]
    cws = cws.t().contiguous()
    qs = qs.t().contiguous()
    idx = torch.searchsorted(cws, qs).t()
    return torch.take_along_dim(input=xs, indices=idx.expand(num_projections, idx.shape[-1]).t().expand(num_dist, idx.shape[-1], num_projections), dim=-2)


num_supports_source = 5
num_supports_target = 4
X = torch.randn(2, num_supports_source, 3)
X_sorted, X_rankings = torch.sort(X, dim=1)
a_cum_weights = torch.linspace(1.0 / num_supports_source, 1.0, steps=num_supports_source)
b_cum_weights = torch.linspace(1.0 / num_supports_target, 1.0, steps=num_supports_target)
qs = torch.sort(torch.concat((a_cum_weights, b_cum_weights), 0), dim=0, descending=False)[0]
print(qs)
print(a_cum_weights)
print(X_sorted)
X_quantiles = quantile_function(qs, a_cum_weights, X_sorted)


# print(qs)
print(X_quantiles)

# x = torch.randn(3)
# y = x.expand(2, 3).t().expand(2, 3, 2)
# print(x)
# print(y)