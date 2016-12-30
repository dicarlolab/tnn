"""
Unicycle Settings
"""

# Default batch size
BATCH_SIZE=256

# Default Harbor_Master policy 
# (  ('max', 'avg', 'up', func) , ('concat','sum',func)  )
HARBOR_MASTER_DEFAULT=('up','concat')