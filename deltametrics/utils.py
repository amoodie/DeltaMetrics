import os
import sys

import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors

from . import io


def _get_version():
    """
    Extract version number from single file, and make it availabe everywhere.
    """
    from . import _version
    return _version.__version__()


def format_number(number):
    integer = int(round(number, -1))
    string = "{:,}".format(integer)
    return(string)


def format_table(number):
    integer = (round(number, 1))
    string = str(integer)
    return(string)


class NoStratigraphyError(AttributeError):
    """Error message for access when no stratigraphy.

    Parameters
    ----------
    obj : :obj:`str`
        Which object user tried to access.

    var : :obj:`str`, optional
        Which variable user tried to access. If provided, more information
        is given in the error message.

    Examples
    --------

    Without the optional `var` argument:

    .. doctest::

        >>> raise utils.NoStratigraphyError(rcm8cube) #doctest: +SKIP
        deltametrics.utils.NoStratigraphyError: 'DataCube' object
        has no preservation or stratigraphy information.

    With the `var` argument given as ``'strat_attr'``:

    .. doctest::

        >>> raise utils.NoStratigraphyError(rcm8cube, 'strat_attr') #doctest: +SKIP
        deltametrics.utils.NoStratigraphyError: 'DataCube' object
        has no attribute 'strat_attr'.
    """

    def __init__(self, obj, var=None):
        """Documented in class docstring."""
        if not (var is None):
            message = "'" + type(obj).__name__ + "'" + " object has no attribute " \
                      "'" + var + "'."
        else:
            message = "'" + type(obj).__name__ + "'" + " object has no preservation " \
                      "or stratigraphy information."
        super().__init__(message)

"""
    yields an exception with:

    .. code::

        deltametrics.utils.NoStratigraphyError: 'DataCube' object
        has no preservation or stratigraphy information.


    .. code::

        deltametrics.utils.NoStratigraphyError: 'DataCube' object
        has no attribute 'strat_attr'.


"""

def needs_stratigraphy(func):
    """Decorator for properties requiring stratigraphy.
    """
    def decorator(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AttributeError as e:
            raise NoStratigraphyError(e)
    return decorator


class AttributeChecker(object):
    """Mixin attribute checker class.

    Registers a method to check whether ``self`` has a given attribute. I.e.,
    the function works as ``hasattr(self, attr)``, where ``attr`` is the
    attribute of interest.

    The benefit of this method over ``hasattr`` is that this method optionally
    takes a list of arguments, and returns a well-formatted error message, to
    help explain which attribute is necessary for a given operation.
    """

    def _attribute_checker(self, checklist):
        """Check for attributes of self.

        Parameters
        ----------
        checklist : `list` of `str`, `str`
            List of attributes to check for existing in ``self``. If a string
            is provided, a single attribute defined by the string is checked
            for. Otherwise, a list of strings is expected, and each string is
            checked.

        .. note::
            This can be refactored to work as a decorator that takes the
            required list as arguments. This could be faster during runtime.
        """

        att_dict = {}
        if type(checklist) is list:
            pass
        elif type(checklist) is str:
            checklist = [checklist]
        else:
            raise TypeError('Checklist must be of type `list`,'
                            'but was type: %s' % type(checklist))

        for c, check in enumerate(checklist):
            has = getattr(self, check, None)
            if has is None:
                att_dict[check] = False
            else:
                att_dict[check] = True

        log_list = [value for value in att_dict.values()]
        log_form = [value for string, value in
                    zip(log_list, att_dict.keys()) if not string]
        if not all(log_list):
            raise RuntimeError('Required attribute(s) not assigned: '
                               + str(log_form))
        return att_dict
