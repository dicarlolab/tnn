"""Path hack to make tests work."""

import os
import sys

bp = os.path.dirname(os.path.realpath('.')).split(os.sep)
modpath = os.sep.join(bp + ['tconvnet'])
# modpath = os.sep.join(bp)
sys.path.insert(0, modpath)
