from typing import *

def calculate_gc_content(dna_sequence):
    """
    Calculate the GC content of the DNA sequence.
    
    Parameters:
    dna_sequence (str): A string containing a sequence of DNA bases.
    
    Returns:
    float: The GC content of the DNA sequence as a percentage, rounded to two decimal places.
    """
    if not dna_sequence:
        return 0.0
    
    total_bases = len(dna_sequence)
    gc_count = dna_sequence.count('G') + dna_sequence.count('C')
    gc_content = (gc_count / total_bases) * 100
    
    return round(gc_content, 2)

### Unit tests below ###
def check(candidate):
    assert candidate("AGCTATAG") == 37.5
    assert candidate("GCGCGCGC") == 100.0
    assert candidate("ATATATAT") == 0.0
    assert candidate("GCATGCAT") == 50.0
    assert candidate("") == 0.0
    assert candidate("A") == 0.0
    assert candidate("G") == 100.0
    assert candidate("ACGTACGT") == 25.0
    assert candidate("CCGGAAGGTT") == 60.0
    assert candidate("TACGCGTACG") == 50.0

def test_check():
    check(calculate_gc_content)
