import numpy as np
import pystrum.pynd.ndutils as nd

def dice_coefficient(output, target, threshold=0.5, smooth=1e-5):
    output = output[:, :, :] > threshold
    target = target[:, :, :] > threshold
    inse = np.count_nonzero(np.logical_and(output, target))
    l = np.count_nonzero(output)
    r = np.count_nonzero(target)
    hard_dice = (2 * inse + smooth) / (l + r + smooth)
    return hard_dice

def jc(disp):
    jc=0
    j_det=jacobian_determinant(disp)
    for i in range(j_det.shape[0]):
        for j in range(j_det.shape[1]):
            for k in range(j_det.shape[2]):
                if j_det[i,j,k] <0:
                    jc=jc+1
    return jc

def jc_proz(disp):
    jc=0
    ges=0
    j_det=jacobian_determinant(disp)
    for i in range(j_det.shape[0]):
        for j in range(j_det.shape[1]):
            for k in range(j_det.shape[2]):
                if j_det[i,j,k] <0:
                    jc=jc+1
                ges=ges+1
    return jc/ges*100

def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
