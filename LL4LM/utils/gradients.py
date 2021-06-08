import logging
import torch
import torch.nn.functional as F
from torch import optim
from copy import deepcopy

import logging
log = logging.getLogger(__name__)

def project(grads, other_grads):
    with torch.no_grad():
        flat_grads = torch.cat([torch.flatten(x) for x in grads])
        flat_other_grads = torch.cat([torch.flatten(x) for x in other_grads])
        dot_product = torch.dot(flat_grads, flat_other_grads)
        if dot_product >= 0:
            return grads
        proj_component = dot_product / torch.dot(flat_other_grads, flat_other_grads)
        proj_grads = [g - proj_component * o for (g, o) in zip(grads, other_grads)]
    return proj_grads

def sequential_gradient_interference(model, prev_grads, prev_nonzero_indices):
    grads, nonzero_indices = get_gradients(model)
    if prev_grads is None or prev_nonzero_indices is None:
        prev_grads = torch.zeros_like(grads).to(grads.device)
        prev_nonzero_indices = torch.zeros_like(nonzero_indices).to(grads.device)
    interference = -F.cosine_similarity(grads, prev_grads, dim=0) 
    overlap = torch.sum(nonzero_indices * prev_nonzero_indices)
    return grads, nonzero_indices, interference, overlap

def pairwise_gradient_similarity(model, names, dataloaders):
    grads, nonzero_mask = {}, {}
    for name, dataloader in zip(names, dataloaders):
        # accumulate gradients
        for i, batch in enumerate(dataloader):
            loss, _ = model.step(batch)
            # scale loss to accumulate the average of gradients
            loss = loss/len(dataloader) 
            loss.backward()
        grads[name], nonzero_mask[name] = get_gradients(model)
        model.zero_grad()
    grad_sim, grad_shared = {}, {}
    for task_i, grad_i in grads.items():
        for task_j, grad_j in grads.items():
            shared_mask = nonzero_mask[task_i] * nonzero_mask[task_j]
            grad_sim[f"grad_sim/{task_i}/{task_j}"] = F.cosine_similarity(
                grad_i[shared_mask], 
                grad_j[shared_mask],
                dim=0
            ).detach().cpu().numpy().item()
            grad_shared[f"grad_shared/{task_i}/{task_j}"] = shared_mask.sum().detach().cpu().numpy().item()
    return grad_sim, grad_shared

def get_gradients(model):
    # extract gradients and indexes of nonzero gradients
    grads, nonzero_mask = [], []
    for _, p in model.named_parameters():
        if p.grad is not None:
            # TODO: clone does not work for some reason
            grad = deepcopy(p.grad.detach().flatten())
            grads.append(grad)
            mask = (grad!=0.0).to(p.device)
            nonzero_mask.append(mask)
        # case for heads of a shared base network
        # where grad will be None
        else:
            shape = p.flatten().shape
            grads.append(torch.zeros(shape).to(p.device))
            nonzero_mask.append(torch.zeros(shape).bool().to(p.device))
    return torch.cat(grads), torch.cat(nonzero_mask)