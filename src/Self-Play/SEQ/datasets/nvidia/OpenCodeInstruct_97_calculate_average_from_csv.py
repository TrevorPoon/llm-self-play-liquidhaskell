from typing import *

def calculate_average_from_csv(file_path, column_name):
    """
    Calculate the average of a specified column in a CSV file.
    
    Parameters:
    file_path (str): The path to the CSV file.
    column_name (str): The name of the column to calculate the average for.
    
    Returns:
    float or None: The average of the column if valid entries are found, otherwise None.
    """
    total = 0
    count = 0
    
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            try:
                value = float(row[column_name])
                total += value
                count += 1
            except (ValueError, KeyError):
                continue
    
    if count == 0:
        return None
    
    return total / count

### Unit tests below ###
def check(candidate):
    assert candidate("data.csv", "Score") == 85.0
    assert candidate("data.csv", "Age") == 28.0
    assert candidate("data.csv", "Name") is None
    assert candidate("data.csv", "NonExistentColumn") is None
    assert candidate("empty.csv", "Score") is None
    assert candidate("data.csv", "Score") == (85 + 92 + 78) / 3
    assert candidate("data.csv", "Age") == (28 + 34 + 22) / 3
    assert candidate("data.csv", "Score") == 85.0
    assert candidate("data.csv", "Age") == 28.0
    assert candidate("data.csv", "Score") == 85.0

def test_check():
    check(calculate_average_from_csv)
