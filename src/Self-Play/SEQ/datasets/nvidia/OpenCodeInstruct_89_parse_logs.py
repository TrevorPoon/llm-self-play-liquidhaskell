from typing import *

def parse_logs(log_entries):
    """
    Parses a list of web server log entries and returns a dictionary mapping each client's IP address
    to the number of requests they have made.

    :param log_entries: List of log entries as strings, where each entry starts with an IP address.
    :return: Dictionary with IP addresses as keys and the count of requests as values.
    """
    ip_counts = {}
    for entry in log_entries:
        ip = entry.split()[0]
        if ip in ip_counts:
            ip_counts[ip] += 1
        else:
            ip_counts[ip] = 1
    return ip_counts

### Unit tests below ###
def check(candidate):
    assert candidate(['192.168.1.1 GET /']) == {'192.168.1.1': 1}
    assert candidate(['192.168.1.1 GET /', '192.168.1.2 POST /login']) == {'192.168.1.1': 1, '192.168.1.2': 1}
    assert candidate(['192.168.1.1 GET /', '192.168.1.1 POST /login', '192.168.1.1 GET /home']) == {'192.168.1.1': 3}
    assert candidate(['192.168.1.1 GET /', '192.168.1.2 POST /login', '192.168.1.3 GET /home']) == {'192.168.1.1': 1, '192.168.1.2': 1, '192.168.1.3': 1}
    assert candidate(['192.168.1.1 GET /', '192.168.1.1 POST /login', '192.168.1.2 GET /home', '192.168.1.2 POST /logout']) == {'192.168.1.1': 2, '192.168.1.2': 2}
    assert candidate([]) == {}
    assert candidate(['10.0.0.1 GET /', '10.0.0.2 POST /login', '10.0.0.1 GET /home', '10.0.0.3 POST /logout', '10.0.0.2 GET /']) == {'10.0.0.1': 2, '10.0.0.2': 2, '10.0.0.3': 1}
    assert candidate(['172.16.0.1 GET /', '172.16.0.1 POST /login', '172.16.0.1 GET /home', '172.16.0.1 POST /logout']) == {'172.16.0.1': 4}
    assert candidate(['192.168.1.1 GET /', '192.168.1.2 POST /login', '192.168.1.1 GET /home', '192.168.1.3 POST /logout', '192.168.1.2 GET /', '192.168.1.3 POST /']) == {'192.168.1.1': 2, '192.168.1.2': 2, '192.168.1.3': 2}
    assert candidate(['192.168.1.1 GET /', '192.168.1.1 POST /login', '192.168.1.1 GET /home', '192.168.1.1 POST /logout', '192.168.1.1 GET /']) == {'192.168.1.1': 5}

def test_check():
    check(parse_logs)
