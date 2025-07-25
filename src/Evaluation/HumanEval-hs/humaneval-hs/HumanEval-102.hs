-- Task ID: HumanEval/102
-- Assigned To: Author D

-- Python Implementation:

--
-- def choose_num(x, y):
--     """This function takes two positive numbers x and y and returns the
--     biggest even integer number that is in the range [x, y] inclusive. If
--     there's no such number, then the function should return -1.
--
--     For example:
--     choose_num(12, 15) = 14
--     choose_num(13, 12) = -1
--     """
--     if x > y:
--         return -1
--     if y % 2 == 0:
--         return y
--     if x == y:
--         return -1
--     return y - 1
--

-- Haskell Implementation:

-- This function takes two positive numbers x and y and returns the
-- biggest even integer number that is in the range [x, y] inclusive. If
-- there's no such number, then the function should return -1.
--
-- For example:
-- choose_num 12 15 = 14
-- choose_num 13 12 = -1
choose_num :: Int -> Int -> Int
choose_num x y
  | x >  y =  -1
  | y `mod`  2 == 0 =  y
  | x ==  y =  -1
  | otherwise =  y - 1
