import torch

def mean_intersection_over_union(mat: torch.Tensor):
    ''' Compute miou via Confmatrix'''
    h = mat.float()
    iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
    miou = torch.mean(iu).item()

    return miou