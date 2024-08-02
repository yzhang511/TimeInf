import abc

ABC = abc.ABC
class AbstractModel(ABC):
    """ Base class. New models can be defined by inheriting from this class
    """

    def __init__(self, *argv, **kwargs):
        """ Initialize a BuiltinUQ object.
        """

    @abc.abstractmethod
    def fit(self, *argv, **kwargs):
        """ Learn model parameters by MLE / MAP fits
        """
        raise NotImplementedError

    @abc.abstractmethod
    def weighted_loss(self, *argv, **kwargs):
        """ Method to compute weighted losses. This is the function we will auto differentiate
            with respect to the weights for IJ computaations
        """
        raise NotImplementedError

