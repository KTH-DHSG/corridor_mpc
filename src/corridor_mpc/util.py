"""
Set of utility functions used across the package.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import casadi as ca


def r_mat(q):
    """
    Generate rotation matrix from unit quaternion

    :param q: unit quaternion
    :type q: ca.MX
    :return: rotation matrix, SO(3)
    :rtype: ca.MX
    """

    Rmat = ca.MX(3, 3)

    # Extract states
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]

    Rmat[0, 0] = 1 - 2 * qy**2 - 2 * qz**2
    Rmat[0, 1] = 2 * qx * qy - 2 * qz * qw
    Rmat[0, 2] = 2 * qx * qz + 2 * qy * qw

    Rmat[1, 0] = 2 * qx * qy + 2 * qz * qw
    Rmat[1, 1] = 1 - 2 * qx**2 - 2 * qz**2
    Rmat[1, 2] = 2 * qy * qz - 2 * qx * qw

    Rmat[2, 0] = 2 * qx * qz - 2 * qy * qw
    Rmat[2, 1] = 2 * qy * qz + 2 * qx * qw
    Rmat[2, 2] = 1 - 2 * qx**2 - 2 * qy**2

    return Rmat


def skew(v):
    """
    Returns the skew matrix of a vector v

    :param v: vector
    :type v: ca.MX
    :return: skew matrix of v
    :rtype: ca.MX
    """

    sk = ca.MX.zeros(3, 3)

    # Extract vector components
    x = v[0]
    y = v[1]
    z = v[2]

    sk[0, 1] = -z
    sk[1, 0] = z
    sk[0, 2] = y
    sk[2, 0] = -y
    sk[1, 2] = -x
    sk[2, 1] = x

    return sk


def inv_skew(sk):
    """
    Retrieve the vector from the skew-symmetric matrix.

    :param sk: skew symmetric matrix
    :type sk: ca.MX
    :return: vector corresponding to SK matrix
    :rtype: ca.MX
    """

    v = ca.MX.zeros(3, 1)

    v[0] = sk[2, 1]
    v[1] = sk[0, 2]
    v[2] = sk[1, 0]

    return v
