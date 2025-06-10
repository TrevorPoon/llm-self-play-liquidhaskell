-- Task ID: HumanEval/122
-- Assigned To: Author D

-- Python Implementation:

-- 
-- def add_elements(arr, k):
--     """
--     Given a non-empty array of integers arr and an integer k, return
--     the sum of the elements with at most two digits from the first k elements of arr.
-- 
--     Example:
-- 
--         Input: arr = [111,21,3,4000,5,6,7,8,9], k = 4
--         Output: 24 # sum of 21 + 3
-- 
--     Constraints:
--         1. 1 <= len(arr) <= 100
--         2. 1 <= k <= len(arr)
--     """
--     return sum(elem for elem in arr[:k] if len(str(elem)) <= 2)
-- 


-- Haskell Implementation:

-- Given a non-empty array of integers arr and an integer k, return
-- the sum of the elements with at most two digits from the first k elements of arr.
-- 
-- Example:
-- 
--     Input: arr = [111,21,3,4000,5,6,7,8,9], k = 4
--     Output: 24 # sum of 21 + 3
-- 
-- Constraints:
--     1. 1 <= len(arr) <= 100
--     2. 1 <= k <= len(arr)
add_elements :: [Int] -> Int -> Int
add_elements arr k =  sum  [x | x <-  take k arr,  length (show x) <= 2]



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (add_elements [1,-2,-3,41,57,76,87,88,99] 3 == -4)
    check (add_elements [111,121,3,4000,5,6] 2 == 0)
    check (add_elements [11,21,3,90,5,6,7,8,9] 4 == 125)
    check (add_elements [111,21,3,4000,5,6,7,8,9] 4 == 24)
    check (add_elements [1] 1 == 1)
