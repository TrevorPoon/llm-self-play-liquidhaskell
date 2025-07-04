-- Task ID: HumanEval/127
-- Assigned To: Author D

-- Python Implementation:

-- 
-- def intersection(interval1, interval2):
--     """You are given two intervals,
--     where each interval is a pair of integers. For example, interval = (start, end) = (1, 2).
--     The given intervals are closed which means that the interval (start, end)
--     includes both start and end.
--     For each given interval, it is assumed that its start is less or equal its end.
--     Your task is to determine whether the length of intersection of these two 
--     intervals is a prime number.
--     Example, the intersection of the intervals (1, 3), (2, 4) is (2, 3)
--     which its length is 1, which not a prime number.
--     If the length of the intersection is a prime number, return "YES",
--     otherwise, return "NO".
--     If the two intervals don't intersect, return "NO".
-- 
-- 
--     [input/output] samples:
--     intersection((1, 2), (2, 3)) ==> "NO"
--     intersection((-1, 1), (0, 4)) ==> "NO"
--     intersection((-3, -1), (-5, 5)) ==> "YES"
--     """
--     def is_prime(num):
--         if num == 1 or num == 0:
--             return False
--         if num == 2:
--             return True
--         for i in range(2, num):
--             if num%i == 0:
--                 return False
--         return True
-- 
--     l = max(interval1[0], interval2[0])
--     r = min(interval1[1], interval2[1])
--     length = r - l
--     if length > 0 and is_prime(length):
--         return "YES"
--     return "NO"
-- 


-- Haskell Implementation:

-- You are given two intervals,
-- where each interval is a pair of integers. For example, interval = (start, end) = (1, 2).
-- The given intervals are closed which means that the interval (start, end)
-- includes both start and end.
-- For each given interval, it is assumed that its start is less or equal its end.
-- Your task is to determine whether the length of intersection of these two 
-- intervals is a prime number.
-- Example, the intersection of the intervals (1, 3), (2, 4) is (2, 3)
-- which its length is 1, which not a prime number.
-- If the length of the intersection is a prime number, return "YES",
-- otherwise, return "NO".
-- If the two intervals don't intersect, return "NO".
-- 
-- 
-- [input/output] samples:
-- intersection (1, 2) (2, 3) ==> "NO"
-- intersection (-1, 1) (0, 4) ==> "NO"
-- intersection (-3, -1) (-5, 5) ==> "YES"
intersection :: (Int, Int) -> (Int, Int) -> String
intersection (a, b) (c, d)
  | b < c || d < a = "NO"
  | otherwise =  if isPrime (abs (min b d - max a c)) then  "YES" else "NO"
  where
    isPrime :: Int -> Bool
    isPrime n =  n > 1 && all (\k -> n `mod` k /= 0) [2..floor (sqrt (fromIntegral n))]
