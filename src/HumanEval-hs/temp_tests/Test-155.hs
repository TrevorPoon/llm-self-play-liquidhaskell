-- Task ID: HumanEval/155
-- Assigned To: Author B

-- Python Implementation:

--
-- def even_odd_count(num):
--     """Given an integer. return a tuple that has the number of even and odd digits respectively.
--
--      Example:
--         even_odd_count(-12) ==> (1, 1)
--         even_odd_count(123) ==> (1, 2)
--     """
--     even_count = 0
--     odd_count = 0
--     for i in str(abs(num)):
--         if int(i)%2==0:
--             even_count +=1
--         else:
--             odd_count +=1
--     return (even_count, odd_count)
--

-- Haskell Implementation:

-- Given an integer. return a tuple that has the number of even and odd digits respectively.
--
-- Example:
-- >>> even_odd_count (-12)
-- (1, 1)
-- >>> even_odd_count 123
-- (1, 2)
even_odd_count :: Int -> (Int, Int)
even_odd_count num =  (length [i | i <-  show (abs num), even (read [i] :: Int)], length [i | i <-  show (abs num), odd (read [i] :: Int)])



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (even_odd_count 7 == (0,1))
    check (even_odd_count (-78) == (1,1))
    check (even_odd_count 3452 == (2,2))
    check (even_odd_count 346211 == (3,3))
    check (even_odd_count (-345821) == (3,3))
    check (even_odd_count (-2) == (1,0))
    check (even_odd_count (-45347) == (2,3))
    check (even_odd_count 0 == (1,0))
