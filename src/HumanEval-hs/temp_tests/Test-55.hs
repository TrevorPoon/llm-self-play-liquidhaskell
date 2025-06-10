-- Task ID: HumanEval/55
-- Assigned To: Author C

-- Python Implementation:

-- 
-- 
-- def fib(n: int):
--     """Return n-th Fibonacci number.
--     >>> fib(10)
--     55
--     >>> fib(1)
--     1
--     >>> fib(8)
--     21
--     """
--     if n == 0:
--         return 0
--     if n == 1:
--         return 1
--     return fib(n - 1) + fib(n - 2)
-- 


-- Haskell Implementation:

-- Return n-th Fibonacci number.
-- >>> fib 10
-- 55
-- >>> fib 1
-- 1
-- >>> fib 8
-- 21
fib :: Int -> Int
fib n =  fib' n 0 1
  where
    fib' :: Int -> Int -> Int -> Int
    fib' 0 a b =  a
    fib' n a b =  fib' (n - 1) b  (a + b)



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (fib 0 == 0)
    check (fib 1 == 1)
    check (fib 2 == 1)
    check (fib 10 == 55)
    check (fib 12 == 144)