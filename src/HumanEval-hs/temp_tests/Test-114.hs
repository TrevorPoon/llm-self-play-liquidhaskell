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
minSubArraySum [x] =  x
minSubArraySum (x:xs) =  minimum  [n, n +  x, x]
  where n =  minSubArraySum  xs


-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (minSubArraySum [2,3,4,1,2,4]           == 1)
    check (minSubArraySum [-1,-2,-3]              == -6)
    check (minSubArraySum [-1,-2,-3,2,-10]       == -14)
    check (minSubArraySum [-9999999999999999]     == -9999999999999999)
    check (minSubArraySum [0,10,20,1000000]       == 0)
    check (minSubArraySum [-1,-2,-3,10,-5]       == -6)
    check (minSubArraySum [100,-1,-2,-3,10,-5]    == -6)
    check (minSubArraySum [10,11,13,8,3,4]        == 3)
    check (minSubArraySum [100,-33,32,-1,0,-2]    == -33)
    check (minSubArraySum [-10]                   == -10)
    check (minSubArraySum [7]                     == 7)
    check (minSubArraySum [1,-1]                  == -1)
