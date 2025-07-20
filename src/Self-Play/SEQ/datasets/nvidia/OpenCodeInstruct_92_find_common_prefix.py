from typing import *

def find_common_prefix(paths):
    """
    Finds the longest common prefix among a list of file paths.
    
    :param paths: List of strings, where each string is a file path.
    :return: A string representing the longest common prefix. Returns an empty string if no common prefix exists.
    """
    if not paths:
        return ""
    
    prefix = paths[0]
    
    for path in paths[1:]:
        while not path.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    
    return prefix

### Unit tests below ###
def check(candidate):
    assert candidate(['/home/user/documents/report.txt', '/home/user/pictures/photo.png', '/home/user/music/song.mp3']) == '/home/user/'
    assert candidate(['/usr/bin/python', '/usr/bin/pip', '/usr/bin/virtualenv']) == '/usr/bin/'
    assert candidate(['/var/log/syslog', '/var/log/auth.log']) == '/var/log/'
    assert candidate(['/etc/nginx/nginx.conf', '/etc/nginx/sites-available/default']) == '/etc/nginx/'
    assert candidate(['/home/user', '/home/user2']) == '/home/'
    assert candidate(['/home/user/documents', '/home/user/documents/report.txt']) == '/home/user/documents'
    assert candidate(['/home/user', '/home/user']) == '/home/user'
    assert candidate([]) == ''
    assert candidate(['/']) == '/'
    assert candidate(['/home/user/documents/report.txt']) == '/home/user/documents/report.txt'

def test_check():
    check(find_common_prefix)
