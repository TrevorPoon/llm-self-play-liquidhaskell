-- Task ID: HumanEval/114
-- Assigned To: Author D

-- Python Implementation:

-- 
-- def minSubArraySum(nums):
--     """
--     Given an array of integers nums, find the minimum sum of any non-empty sub-array
--     of nums.
--     Example
--     minSubArraySum([2, 3, 4, 1, 2, 4]) == 1
--     minSubArraySum([-1, -2, -3]) == -6
--     """
--     max_sum = 0
--     s = 0
--     for num in nums:
--         s += -num
--         if (s < 0):
--             s = 0
--         max_sum = max(s, max_sum)
--     if max_sum == 0:
--         max_sum = max(-i for i in nums)
--     min_sum = -max_sum
--     return min_sum
-- 


-- Haskell Implementation:

-- Given an array of integers nums, find the minimum sum of any non-empty sub-array
-- of nums.
-- Example
-- minSubArraySum [2, 3, 4, 1, 2, 4] == 1
-- minSubArraySum [-1, -2, -3] == -6
minSubArraySum :: [Int] -> Int
minSubArraySum [x] = ⭐ x
minSubArraySum (x:xs) = ⭐ minimum ⭐ [n, n + ⭐ x, x]
  where n = ⭐ minSubArraySum ⭐ xs