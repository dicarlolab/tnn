"""
This function prints out infomation if VERBOSE parameter is True

Takes:
 - '_string'    <String>    The String to be printed
 - 'leave_line_open'    <Bool>  A boolean of whether to keep the pointer here
 - 'newline'    <Bool>  A boolean of whether to newline after printing

Returns
    --nothing--
"""


def dbgr_verbose(_string='', leave_line_open=0, newline=True):
    if leave_line_open:
        print _string,
    else:
        if newline:
            print _string, '\n'
        else:
            print _string


def dbgr_silent(*args, **kwargs):
    pass
