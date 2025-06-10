-- Task ID: HumanEval/97
-- Assigned To: Author E

-- Python Implementation:

--
-- def multiply(a, b):
--     """Complete the function that takes two integers and returns
--     the product of their unit digits.
--     Assume the input is always valid.
--     Examples:
--     multiply(148, 412) should return 16.
--     multiply(19, 28) should return 72.
--     multiply(2020, 1851) should return 0.
--     multiply(14,-15) should return 20.
--     """
--     return abs(a % 10) * abs(b % 10)
--

-- Haskell Implementation:

-- Complete the function that takes two integers and returns
-- the product of their unit digits.
-- Assume the input is always valid.
-- Examples:
-- multiply 148 412 should return 16.
-- multiply 19 28 should return 72.
-- multiply 2020 1851 should return 0.
-- multiply 14 (-15) should return 20.

multiply :: Int -> Int -> Int
multiply a b =
  abs  (a `mod` 10)
    * abs  (b `mod` 10)

-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (multiply 148 412 == 16)
    check (multiply 19 28 == 72)
    check (multiply 2020 1851 == 0)
    check (multiply 14 (-15) == 20)
    check (multiply 76 67 == 42)
    check (multiply 17 27 == 49)
    check (multiply 0 1 == 0)
    check (multiply 0 0 == 0)
