-- Task ID: HumanEval/73
-- Assigned To: Author E

-- Python Implementation:

-- 
-- def smallest_change(arr):
--     """
--     Given an array arr of integers, find the minimum number of elements that
--     need to be changed to make the array palindromic. A palindromic array is an array that
--     is read the same backwards and forwards. In one change, you can change one element to any other element.
-- 
--     For example:
--     smallest_change([1,2,3,5,4,7,9,6]) == 4
--     smallest_change([1, 2, 3, 4, 3, 2, 2]) == 1
--     smallest_change([1, 2, 3, 2, 1]) == 0
--     """
--     ans = 0
--     for i in range(len(arr) // 2):
--         if arr[i] != arr[len(arr) - i - 1]:
--             ans += 1
--     return ans
-- 


-- Haskell Implementation:

-- Given an array arr of integers, find the minimum number of elements that
-- need to be changed to make the array palindromic. A palindromic array is an array that
-- is read the same backwards and forwards. In one change, you can change one element to any other element.
--
-- For example:
-- smallest_change [1,2,3,5,4,7,9,6] == 4
-- smallest_change [1, 2, 3, 4, 3, 2, 2] == 1
-- smallest_change [1, 2, 3, 2, 1] == 0

smallest_change :: [Int] -> Int
smallest_change arr = ⭐ length [i | i <- [0..(length arr `div` 2 - 1)], arr !! i /= arr !! ⭐ (length arr - i - 1)]
