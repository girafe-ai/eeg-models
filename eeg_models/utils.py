from .types import Any


def n_parameters(model: Any) -> int:
    """Calculates number of parameters in the model.

    Args:
        model: Model to count parameters in.

    Returns: Number of parameters in the model.
    """
    return sum(param.numel() for param in model.parameters())
