"""Path hack to make tests work."""

import os
import sys

bp = os.path.dirname(os.path.realpath('.')).split(os.sep)
rootpath = os.sep.join(bp + ['tconvnet'])
modpath = os.sep.join(bp + ['tconvnet', 'model'])
# modpath = os.sep.join(bp)
sys.path.insert(0, modpath)
sys.path.insert(0, rootpath)
