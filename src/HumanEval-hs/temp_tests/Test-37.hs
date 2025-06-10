-- Task ID: HumanEval/37
-- Assigned To: Author C

-- Python Implementation:

-- 
-- 
-- def sort_even(l: list):
--     """This function takes a list l and returns a list l' such that
--     l' is identical to l in the odd indicies, while its values at the even indicies are equal
--     to the values of the even indicies of l, but sorted.
--     >>> sort_even([1, 2, 3])
--     [1, 2, 3]
--     >>> sort_even([5, 6, 3, 4])
--     [3, 6, 5, 4]
--     """
--     evens = l[::2]
--     odds = l[1::2]
--     evens.sort()
--     ans = []
--     for e, o in zip(evens, odds):
--         ans.extend([e, o])
--     if len(evens) > len(odds):
--         ans.append(evens[-1])
--     return ans
-- 


-- Haskell Implementation:
import Data.List (sort)

-- This function takes a list l and returns a list l' such that
-- l' is identical to l in the odd indicies, while its values at the even indicies are equal
-- to the values of the even indicies of l, but sorted.
-- >>> sort_even [1,2,3]
-- [1,2,3]
-- >>> sort_even [5,6,3,4]
-- [3,6,5,4]
sort_even :: [Int] -> [Int]
sort_even xs = replaceEverySecond xs (sort $ everySecond xs)
  where
    everySecond :: [Int] -> [Int]
    everySecond [] =  []
    everySecond (x:xs) =  x : everySecond  (drop 1 xs)
    replaceEverySecond :: [Int] -> [Int] -> [Int]
    replaceEverySecond [] _ =  []
    replaceEverySecond xs [] =  xs
    replaceEverySecond (x:xs) (y:ys) =  y : (take 1 xs ++  replaceEverySecond  (drop 1 xs) ys)



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (sort_even [1,2,3] == [1,2,3])
    check (sort_even [5,3,-5,2,-3,3,9,0,123,1,-10] == [-10,3,-5,2,-3,3,5,0,9,1,123])
    check (sort_even [5,8,-12,4,23,2,3,11,12,-10] == [-12,8,3,4,5,2,12,11,23,-10])
