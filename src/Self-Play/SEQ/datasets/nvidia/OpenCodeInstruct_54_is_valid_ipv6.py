from typing import *

def is_valid_ipv6(ipv6_address):
    """
    Validates if the given string is a valid IPv6 address.

    :param ipv6_address: A string representing an IPv6 address.
    :return: True if the string is a valid IPv6 address, False otherwise.
    """
    ipv6_pattern = re.compile('''
        ^  # start of string
        (?:
            (?:  # non-capturing group for 7 groups of 4 hex digits separated by colons
                [0-9A-Fa-f]{1,4}:  # 1 to 4 hex digits followed by a colon
            ){0,7}  # zero to seven repetitions
            (?:  # non-capturing group for the final group of 4 hex digits
                [0-9A-Fa-f]{1,4}  # 1 to 4 hex digits
            )  # end of final group
            |  # OR
            (?:  # non-capturing group for IPv6 addresses with double colons
                (?:  # non-capturing group for 1 to 6 groups of 4 hex digits separated by colons
                    [0-9A-Fa-f]{1,4}:  # 1 to 4 hex digits followed by a colon
                ){0,1}  # zero or one repetition
                :  # double colon
                (?:  # non-capturing group for 0 to 5 groups of 4 hex digits separated by colons
                    [0-9A-Fa-f]{1,4}:  # 1 to 4 hex digits followed by a colon
                ){0,5}  # zero to five repetitions
                (?:  # non-capturing group for the final group of 4 hex digits
                    [0-9A-Fa-f]{1,4}  # 1 to 4 hex digits
                )  # end of final group
                |  # OR
                (?:  # non-capturing for IPv6 addresses ending in double colon
                    [0-9A-Fa-f]{1,4}:  # 1 to 4 hex digits followed by a colon
                ){1,7}  # one to seven repetitions
                :  # single colon at the end
            )  # end of IPv6 addresses with double colons
        )  # end of main non-capturing group
        $  # end of string
    ''', re.VERBOSE)
    return ipv6_pattern.match(ipv6_address) is not None

### Unit tests below ###
def check(candidate):
    assert ipv6_pattern.match('2001:0db8:85a3:0000:0000:8a2e:0370:7334') is not None
    assert ipv6_pattern.match('2001:db8:85a3::8a2e:370:7334') is not None
    assert ipv6_pattern.match('::') is not None
    assert ipv6_pattern.match('2001:db8::') is not None
    assert ipv6_pattern.match('2001::1') is not None
    assert ipv6_pattern.match('2001:db8::8a2e:370:7334') is not None
    assert ipv6_pattern.match('2001:db8:85a3:0:0:8a2e:370:7334') is not None
    assert ipv6_pattern.match('2001:db8::85a3:0:0:8a2e:370:7334') is None
    assert ipv6_pattern.match('2001:db8::1:2:3::4') is None
    assert ipv6_pattern.match('2001:db8:85a3::12345') is None

def test_check():
    check(is_valid_ipv6)
