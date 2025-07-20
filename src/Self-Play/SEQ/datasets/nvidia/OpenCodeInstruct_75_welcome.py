from typing import *

def welcome():
    """
    Returns a welcome message when the root endpoint is accessed.
    
    Returns:
        str: A welcome message.
    """
    return 'Welcome to the Simple Flask Server!'

### Unit tests below ###
def check(candidate):
    assert app.name == 'flask.app'
    assert candidate() == 'Welcome to the Simple Flask Server!'
    assert 'Current Time:' in current_time()
    assert len(current_time().split(':')) == 3
    assert current_time().split()[1].count('-') == 2
    assert current_time().split()[3].count(':') == 2
    assert app.url_map.bind('').match('/') == ('welcome', {})
    assert app.url_map.bind('').match('/time') == ('current_time', {})
    assert app.url_map.is_endpoint_expecting('welcome', 'arg') == False
    assert app.url_map.is_endpoint_expecting('current_time', 'arg') == False

def test_check():
    check(welcome)
