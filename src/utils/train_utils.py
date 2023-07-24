def l2_regularization(m):
    """
    Calculates the L2 regularization term for a given model. It adds a
    penalty term to the loss function that encourages smaller weight values.
    This function iterates over the parameters of the provided model and
    computes the L2 norm of each parameter tensor. The L2 norms are summed
    to obtain the overall L2 regularization term.

    Parameters
    ----------
    m : nn.Module
        The model for which the L2 regularization term will be computed.

    Returns
    -------
    l2_reg : torch.Tensor or None
        The L2 regularization term for the model.

    """
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)

    return l2_reg
