-- Task ID: HumanEval/20
-- Assigned To: Author A

-- Python Implementation:

-- from typing import List, Tuple
-- 
-- 
-- def find_closest_elements(numbers: List[float]) -> Tuple[float, float]:
--     """ From a supplied list of numbers (of length at least two) select and return two that are the closest to each
--     other and return them in order (smaller number, larger number).
--     >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2])
--     (2.0, 2.2)
--     >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0])
--     (2.0, 2.0)
--     """
--     closest_pair = None
--     distance = None
-- 
--     for idx, elem in enumerate(numbers):
--         for idx2, elem2 in enumerate(numbers):
--             if idx != idx2:
--                 if distance is None:
--                     distance = abs(elem - elem2)
--                     closest_pair = tuple(sorted([elem, elem2]))
--                 else:
--                     new_distance = abs(elem - elem2)
--                     if new_distance < distance:
--                         distance = new_distance
--                         closest_pair = tuple(sorted([elem, elem2]))
-- 
--     return closest_pair
-- 


-- Haskell Implementation:
import Data.List
import Data.Ord   (comparing)
import Data.List  (minimumBy)

-- From a supplied list of numbers (of length at least two) select and return two that are the closest to each
-- other and return them in order (smaller number, larger number).
-- >>> find_closest_elements [1.0, 2.0, 3.0, 4.0, 5.0, 2.2]
-- (2.0,2.2)
-- >>> find_closest_elements [1.0, 2.0, 3.0, 4.0, 5.0, 2.0]
-- (2.0,2.0)
find_closest_elements :: (Ord a, Num a) => [a] -> (a, a)
find_closest_elements xs
  | length xs < 2 = error "need at least two elements"
  | otherwise     = minimumBy (comparing (\(a,b) -> b - a)) pairs
  where
    sorted = sort xs
    pairs  = zip sorted (tail sorted)
