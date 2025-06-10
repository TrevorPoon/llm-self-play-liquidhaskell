-- Task ID: HumanEval/150
-- Assigned To: Author B

-- Python Implementation:

--
-- def x_or_y(n, x, y):
--     """A simple program which should return the value of x if n is
--     a prime number and should return the value of y otherwise.
--
--     Examples:
--     for x_or_y(7, 34, 12) == 34
--     for x_or_y(15, 8, 5) == 5
--
--     """
--     if n == 1:
--         return y
--     for i in range(2, n):
--         if n % i == 0:
--             return y
--             break
--     else:
--         return x
--

-- Haskell Implementation:

-- A simple program which should return the value of x if n is
-- a prime number and should return the value of y otherwise.
--
-- Examples:
-- >>> x_or_y 7 34 12
-- 34
-- >>> x_or_y 15 8 5
-- 5
x_or_y :: Int -> Int -> Int -> Int
x_or_y n x y =  if n == 1 then y else if length [i |  i <- [2 .. n - 1], n `mod` i == 0] > 0 then  y else x



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (x_or_y 7 34 12 == 34)
    check (x_or_y 15 8 5 == 5)
    check (x_or_y 3 33 5212 == 33)
    check (x_or_y 1259 3 52 == 3)
    check (x_or_y 7919 (-1) 12 == -1)
    check (x_or_y 3609 1245 583 == 583)
    check (x_or_y 91 56 129 == 129)
    check (x_or_y 6 34 1234 == 1234)
    check (x_or_y 1 2 0 == 0)
    check (x_or_y 2 2 0 == 2)
