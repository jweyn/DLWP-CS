#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Base re-mapping tools.
"""

import os
import sys


class _BaseRemap(object):
    """
    Base class for remappers. Implements basic functions.
    """

    def __init__(self, path_to_remapper=None):
        if path_to_remapper is None:
            self.path_to_remapper = os.path.dirname(sys.executable)
        else:
            self.path_to_remapper = path_to_remapper
        pass

    def remap(self, input_file, output_file):
        """
        Forward mapping operation. Must be defined in subclasses.
        """
        pass

    def inverse_remap(self, input_file, output_file):
        """
        Inverse mapping operation. Must be defined in subclasses.
        """
        pass
