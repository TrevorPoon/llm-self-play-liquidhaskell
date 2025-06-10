-- Task ID: HumanEval/63
-- Assigned To: Author C

-- Python Implementation:

-- 
-- 
-- def fibfib(n: int):
--     """The FibFib number sequence is a sequence similar to the Fibbonacci sequnece that's defined as follows:
--     fibfib(0) == 0
--     fibfib(1) == 0
--     fibfib(2) == 1
--     fibfib(n) == fibfib(n-1) + fibfib(n-2) + fibfib(n-3).
--     Please write a function to efficiently compute the n-th element of the fibfib number sequence.
--     >>> fibfib(1)
--     0
--     >>> fibfib(5)
--     4
--     >>> fibfib(8)
--     24
--     """
--     if n == 0:
--         return 0
--     if n == 1:
--         return 0
--     if n == 2:
--         return 1
--     return fibfib(n - 1) + fibfib(n - 2) + fibfib(n - 3)
-- 


-- Haskell Implementation:

-- The FibFib number sequence is a sequence similar to the Fibbonacci sequnece that's defined as follows:
-- fibfib(0) == 0
-- fibfib(1) == 0
-- fibfib(2) == 1
-- fibfib(n) == fibfib(n-1) + fibfib(n-2) + fibfib(n-3).
-- Please write a function to efficiently compute the n-th element of the fibfib number sequence.
-- >>> fibfib 1
-- 0
-- >>> fibfib 5
-- 4
-- >>> fibfib 8
-- 24
fibfib :: Int -> Int
fibfib n
  | n == 0 =  0
  | n == 1 =  0
  | n == 2 =  1
  | otherwise =  fibfib (n - 1) + fibfib (n - 2) +  fibfib (n - 3)



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (fibfib 1 == 0)
    check (fibfib 2 == 1)
    check (fibfib 5 == 4)
    check (fibfib 8 == 24)
    check (fibfib 10 == 81)
    check (fibfib 12 == 274)
    check (fibfib 14 == 927)