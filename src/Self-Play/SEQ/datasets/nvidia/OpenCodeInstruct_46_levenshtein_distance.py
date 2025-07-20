from typing import *

def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    
    :param s1: First string
    :param s2: Second string
    :return: Levenshtein distance between s1 and s2
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

### Unit tests below ###
def check(candidate):
    assert closest_match(["apple", "apply", "ample"], "appel") == "apple"
    assert closest_match(["kitten", "sitting", "kitchen"], "kitten") == "kitten"
    assert closest_match(["flaw", "lawn", "flawed"], "lawn") == "lawn"
    assert closest_match(["intention", "execution", "intentional"], "execution") == "execution"
    assert closest_match(["algorithm", "altruism", "algorithmic"], "algorithm") == "algorithm"
    assert closest_match(["distance", "difference", "dissimilar"], "distance") == "distance"
    assert closest_match(["levenshtein", "levenschtein", "levinsthein"], "levenshtein") == "levenshtein"
    assert closest_match(["hello", "hallo", "hullo"], "hallo") == "hallo"
    assert closest_match(["programming", "programing", "progrmming"], "programming") == "programming"
    assert closest_match(["test", "tset", "sett"], "test") == "test"

def test_check():
    check(levenshtein_distance)
