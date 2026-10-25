import torch


def distributed_sinkhorn_balanced(out, sinkhorn_iterations=50, epsilon=0.05):
    """Return the balanced soft assignment and its deterministic hard indices."""
    if out.ndim != 2:
        raise ValueError(f'Expected a 2D similarity matrix, got shape {tuple(out.shape)}')
    if out.shape[0] == 0 or out.shape[1] == 0:
        raise ValueError(f'Sinkhorn requires a non-empty matrix, got shape {tuple(out.shape)}')

    with torch.no_grad():
        logits = out.float() / epsilon
        logits = (logits - logits.max()).clamp_min(-50.0)
        assignment = torch.exp(logits).t()
        num_samples = assignment.shape[1]
        num_prototypes = assignment.shape[0]
        tiny = torch.finfo(assignment.dtype).tiny

        assignment /= assignment.sum().clamp_min(tiny)
        for _ in range(sinkhorn_iterations):
            assignment /= assignment.sum(dim=1, keepdim=True).clamp_min(tiny)
            assignment /= num_prototypes
            assignment /= assignment.sum(dim=0, keepdim=True).clamp_min(tiny)
            assignment /= num_samples

        assignment *= num_samples
        assignment = assignment.t().to(dtype=out.dtype)
        indices = assignment.argmax(dim=1)

    return assignment, indices
