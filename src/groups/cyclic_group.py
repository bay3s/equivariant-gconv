import numpy as np
import torch

from . import BaseGroup


class CyclicGroup(BaseGroup):
  """
  This class implements the C4 Cyclic group which is the group containing all 90-degree rotations of the plane.

  - The set of group elements of C4 is given by G = {e, g1, g2, g3}.
  - We can porameterize these group elements using the rotation angle "theta" (ie. e = 0, g = 0.5pi, g2 = pi, ..)
  - The inverse is given by g^{-1} = -Theta mod (2pi)
  - The group has an action on the Euclidean plan in dimensions R2 given by a rotation matrix
    R(theta) = [[cos(theta), -sin(theta)], [sin(theta), cost(theta)]]
  - The regular representation of functions F defined over R2 is f(R(-theta) mod 2pi.x)
  """

  def __init__(self, order: int):
    """
    Initialize the C4 Cyclic group.

    Args:
      order (int): The order of the group.
    """
    super().__init__(
      dimension = 1,
      identity = [0.]
    )

    assert order > 1
    self.order = torch.tensor(order)
    pass

  def elements(self) -> torch.Tensor:
    """
    Returns a tensor containing all group elements.

    Returns:
      torch.Tensor
    """
    return torch.linspace(
      start = 0,
      end = 2 * np.pi * float(self.order - 1) / float(self.order),
      steps = self.order,
      device = self.identity.device
    )

  def product(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
    """
    Defines a group product on two group elements of the cyclic group C4.

    Args:
      g1 (torch.Tensor): The first group element.
      g2 (torch.Tensor): The second group element.

    Returns:
      torch.Tensor
    """
    return torch.remainder(g1 + g2, 2 * np.pi)

  def inverse(self, g: torch.Tensor) -> torch.Tensor:
    """
    Defines the group inverse for an element of the Cycli Group C4.

    Args:
      g (torch.Tensor):

    Returns:

    """
    return torch.remainder(-g, 2 * np.pi)

  def left_action_on_R2(self, batch_g, batch_x) -> torch.Tensor:
    """
    Group action of an element g on a set of vectors in R2.

    Args:
      batch_g (torch.Tensor): Batch of group elements.
      batch_x (torch.Tensor): Tensor of vectors in R2.

    Returns:
      torch.Tensor
    """

    """
    Create a tensor containing representations of each of the group elements in the input. 
    Creates a tensor of size [batch_size, 2, 2].
    """
    batched_rep = torch.stack([self.matrix_representation(h) for h in batch_g])

    """
    Transform the R2 input grid with each representation to end up with a transformed grid of dimensionality 
    [num_group_elements, spatial_dim_y, spatial_dim_x, 2].
    """
    out = torch.einsum('boi,ixy->bxyo', batched_rep, batch_x)

    """
    Afterwards (because grid_sample assummes our grid is y,x instead of x,y) we swap x and y coordinate values 
    with a roll along final dimension.
    """
    return out.roll(shifts = 1, dims = -1)

  def left_action_on_H(self, batch_h, batch_h_prime):
    """
    Group action of an element h on a set of group elements in H, nothing more than a batch-wise group product.

    The elements in batch_g work on the elements in batch_h_prime directly, through the group product.

    Each element in batch_g is applied to each element in batch_h_prime.

    Args:
      batch_h (): A tensor containing group elements.
      batch_h_prime (): A tensor of group elements to apply the group product to.

    Returns:
      torch.Tensor
    """
    return self.product(batch_h.repeat(batch_h_prime.shape[0], 1), batch_h_prime.unsqueeze(-1))

  def matrix_representation(self, g: torch.Tensor) -> torch.Tensor:
    """
    Obtain a matrix representation in R^2 for an element g.

    Args:
      g (torch.Tensor): A group element.

    Returns:
      torch.Tensor
    """
    cos_t = torch.cos(g)
    sin_t = torch.sin(g)

    return torch.tensor([
      [cos_t, -sin_t],
      [sin_t, cos_t]
    ], device = self.identity.device)

  def normalize_group_elements(self, g: torch.Tensor) -> torch.Tensor:
    """
    Normalize values of group elements to range between -1 and 1.

    The group elements range from 0 to 2pi * (self.order - 1) / self.order, so we normalize by the largest element
    in the group.

    Args:
      g (torch.Tensor): A group element.

    Returns:
      torch.Tensor
    """
    largest_elem = 2 * np.pi * (self.order - 1) / self.order

    return (2 * g / largest_elem) - 1.

  def determininant(self, h):
    pass
