import os
import pkg_resources

MXM2MSD = 'data/mxm2msd.txt'


__all__ = [
    'mxm2msd'
]


def mxm2msd():
    """ Read the filename of map between MxM and MSD

    Returns:
        str: filename of map between MxM and MSD
    """
    return pkg_resources.resource_filename(__name__, MXM2MSD)
