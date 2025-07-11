-- Task ID: HumanEval/19
-- Assigned To: Author A

-- Python Implementation:

-- from typing import List
-- 
-- 
-- def sort_numbers(numbers: str) -> str:
--     """ Input is a space-delimited string of numberals from 'zero' to 'nine'.
--     Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
--     Return the string with numbers sorted from smallest to largest
--     >>> sort_numbers('three one five')
--     'one three five'
--     """
--     value_map = {
--         'zero': 0,
--         'one': 1,
--         'two': 2,
--         'three': 3,
--         'four': 4,
--         'five': 5,
--         'six': 6,
--         'seven': 7,
--         'eight': 8,
--         'nine': 9
--     }
--     return ' '.join(sorted([x for x in numbers.split(' ') if x], key=lambda x: value_map[x]))
-- 


-- Haskell Implementation:
import Data.Map
import Data.List

-- Input is a space-delimited string of numberals from 'zero' to 'nine'.
-- Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
-- Return the string with numbers sorted from smallest to largest
-- >>> sort_numbers "three one five"
-- "one three five"
sort_numbers :: String -> String
sort_numbers numbers =  unwords $ sortOn  (value_map !) $  words numbers
    where value_map = fromList  [("zero", 0), ("one", 1), ("two", 2), ("three", 3), ("four", 4), ("five", 5),  ("six", 6), ("seven", 7), ("eight", 8), ("nine", 9)]
