-- Task ID: HumanEval/39
-- Assigned To: Author C

-- Python Implementation:

-- 
-- 
-- def prime_fib(n: int):
--     """
--     prime_fib returns n-th number that is a Fibonacci number and it's also prime.
--     >>> prime_fib(1)
--     2
--     >>> prime_fib(2)
--     3
--     >>> prime_fib(3)
--     5
--     >>> prime_fib(4)
--     13
--     >>> prime_fib(5)
--     89
--     """
--     import math
-- 
--     def is_prime(p):
--         if p < 2:
--             return False
--         for k in range(2, min(int(math.sqrt(p)) + 1, p - 1)):
--             if p % k == 0:
--                 return False
--         return True
--     f = [0, 1]
--     while True:
--         f.append(f[-1] + f[-2])
--         if is_prime(f[-1]):
--             n -= 1
--         if n == 0:
--             return f[-1]
-- 


-- Haskell Implementation:

-- prime_fib returns n-th number that is a Fibonacci number and it's also prime.
-- >>> prime_fib 1
-- 2
-- >>> prime_fib 2
-- 3
-- >>> prime_fib 3
-- 5
-- >>> prime_fib 4
-- 13
-- >>> prime_fib 5
-- 89
prime_fib :: Int -> Int
prime_fib n = primeFibs !! (n - 1)
  where
    -- 1,1,2,3,5,8,13,21,34,55,...
    fibs :: [Int]
    fibs = 1 : 1 : zipWith (+) fibs (tail fibs)

    -- filter that down to primes
    primeFibs :: [Int]
    primeFibs = filter isPrime fibs

    isPrime :: Int -> Bool
    isPrime p
      | p < 2     = False
      | otherwise = null [ k | k <- [2..limit], p `mod` k == 0 ]
      where
        limit = floor . sqrt $ fromIntegral p

