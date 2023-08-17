'''This module contains various regular expression patterns

Each pattern can be used as an argument to the re.compile() function.
'''

def casrn(as_group=False):
    '''Return regex pattern for matching a chemical's CAS Registry Number 
    (CASRN).

    CASRN consists of 3 parts separated by hyphens: 
        \d{2,7}-   1st part (2-7 digits)
        \d{2}-     2nd part (2 digits)
        \d         3rd part (1 digit)
    '''
    pat = r'\d{2,7}-\d{2}-\d'
    if as_group is True:
        pat = group_pattern(pat)
    return pat

def dtxsid(as_group=False):
    '''Return regex pattern for matching a chemical's DSSTox substance identifier 
    (DTXSID) used in the U.S. EPA CompTox Dashboard.
    '''
    pat = r'DTXSID\d{7,9}'
    if as_group is True:
        pat = group_pattern(pat)
    return pat

def element(as_group=False):
    '''Return regex pattern for matching an element in the periodic table.

    An element can be expressed as a capital letter followed by zero or one
    lowercase letters.
    '''
    pat = r'[A-Z][a-z]?'
    if as_group is True:
        pat = group_pattern(pat)
    return pat

def group_pattern(pat):
    '''Return the regex pattern (str) as a single group.
    '''
    return '(' + pat + ')'