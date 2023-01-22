from abc import ABC, abstractmethod
import torch


class BaseGroup(ABC, torch.nn.Module):

  def __init__(self, dimension, identity):
    """
    Initialize the group.

    Args:
      dimension ():
      identity ():
    """
    super().__init__()
    self.dimension = dimension
    self.register_buffer('identity', tensor = torch.Tensor(identity))
    pass

  @abstractmethod
  def elements(self) -> torch.Tensor:
    """
    Returns a tensor containing all group elements.
    """
    raise NotImplementedError()

  def product(self, h, h_prime):
    """
    Defines the group product on two group elements.

    Args:
      h ():
      h_prime ():

    Returns:

    """
    raise NotImplementedError()

  def left_action_on_R2(self, h_batch, x_batch):
    """
    Group action on an element from the subgroup H on a vector in R2.

    Args:
      h_batch ():
      x_batch ():

    Returns:

    """
    raise NotImplementedError()

  def left_action_on_H(self, h_batch, h_prime_batch):
    """
    Group action of elements of H on other elements in H itself - comes down to group product.

    For efficiency we implement this batchwise - each element in h_batch is applied to each element in h_prime_batch.

    Args:
      h_batch ():
      h_prime_batch ():

    Returns:

    """
    raise NotImplementedError()

  def matrix_representation(self, h):
    """
    Obtain a matrix representation in R^2 of a group element h.

    Args:
      h ():

    Returns:

    """
    raise NotImplementedError()

  def determininant(self, h):
    """
    Compute the determinant of the represetnation of a group element h.

    Args:
      self ():
      h ():

    Returns:

    """
    raise NotImplementedError()

  def normalize_group_elements(self, h):
    """
    Map the group elements to an interval [-1, 1]

    We use this to create a standardized input for obtaining weights over the group.

    Args:
      h ():

    Returns:

    """
