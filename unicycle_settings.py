"""
Unicycle Settings
"""

# Default batch size
BATCH_SIZE=256

# Verbose mode on or off
VERBOSE=True

# Default Harbor_Master policy 
# (  ('max', 'avg', 'up', func) , ('concat','sum',func)  )
# HARBOR_MASTER_DEFAULT=('up','concat')

# New Default Harbor Master Policy is a function that takes in a collection
# of input nodes and returns an 
def harbor_master_default():
	print 'This is the harbor_master_default policy function being run'