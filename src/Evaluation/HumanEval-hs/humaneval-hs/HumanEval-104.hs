-- Task ID: HumanEval/104
-- Assigned To: Author D

-- Python Implementation:

--
-- def unique_digits(x):
--     """Given a list of positive integers x. return a sorted list of all
--     elements that hasn't any even digit.
--
--     Note: Returned list should be sorted in increasing order.
--
--     For example:
--     >>> unique_digits([15, 33, 1422, 1])
--     [1, 15, 33]
--     >>> unique_digits([152, 323, 1422, 10])
--     []
--     """
--     odd_digit_elements = []
--     for i in x:
--         if all (int(c) % 2 == 1 for c in str(i)):
--             odd_digit_elements.append(i)
--     return sorted(odd_digit_elements)
--

-- Haskell Implementation:

-- Given a list of positive integers x. return a sorted list of all
-- elements that hasn't any even digit.
--
-- Note: Returned list should be sorted in increasing order.
--
-- For example:
-- >>> unique_digits [15, 33, 1422, 1]
-- [1,15,33]
-- >>> unique_digits [152, 323, 1422, 10]
-- []

import Data.Char (digitToInt)
import Data.List (sort)

unique_digits :: [Int] -> [Int]
unique_digits x =  sort $ filter
      (all odd .  map digitToInt .  show)
      x
