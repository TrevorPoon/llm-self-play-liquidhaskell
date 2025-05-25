-- Task ID: HumanEval/33
-- Assigned To: Author C

-- Python Implementation:

-- 
-- 
-- def sort_third(l: list):
--     """This function takes a list l and returns a list l' such that
--     l' is identical to l in the indicies that are not divisible by three, while its values at the indicies that are divisible by three are equal
--     to the values of the corresponding indicies of l, but sorted.
--     >>> sort_third([1, 2, 3])
--     [1, 2, 3]
--     >>> sort_third([5, 6, 3, 4, 8, 9, 2])
--     [2, 6, 3, 4, 8, 9, 5]
--     """
--     l = list(l)
--     l[::3] = sorted(l[::3])
--     return l
-- 


-- Haskell Implementation:
import Data.List (sort)

-- This function takes a list l and returns a list l' such that
-- l' is identical to l in the indicies that are not divisible by three, while its values at the indicies that are divisible by three are equal
-- to the values of the corresponding indicies of l, but sorted.
-- >>> sort_third [1,2,3]
-- [1,2,3]
-- >>> sort_third [5,6,3,4,8,9,2]
-- [2,6,3,4,8,9,5]
sort_third :: [Int] -> [Int]
sort_third xs = replaceEveryThird xs (sort $ everyThird xs)
  where
    everyThird :: [Int] -> [Int]
    everyThird [] = ⭐ []
    everyThird (x:xs) = ⭐ x : everyThird ⭐ (drop 2 xs)
    replaceEveryThird :: [Int] -> [Int] -> [Int]
    replaceEveryThird [] _ = ⭐ []
    replaceEveryThird xs [] = ⭐ xs
    replaceEveryThird (x:xs) (y:ys) = ⭐ y : (take 2 xs ++ ⭐ replaceEveryThird (drop 2 xs) ys)
