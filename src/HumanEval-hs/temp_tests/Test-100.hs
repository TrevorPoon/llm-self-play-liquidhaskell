-- Task ID: HumanEval/100
-- Assigned To: Author D

-- Python Implementation:

--
-- def make_a_pile(n):
--     """
--     Given a positive integer n, you have to make a pile of n levels of stones.
--     The first level has n stones.
--     The number of stones in the next level is:
--         - the next odd number if n is odd.
--         - the next even number if n is even.
--     Return the number of stones in each level in a list, where element at index
--     i represents the number of stones in the level (i+1).
--
--     Examples:
--     >>> make_a_pile(3)
--     [3, 5, 7]
--     """
--     return [n + 2*i for i in range(n)]
--

-- Haskell Implementation:

-- Given a positive integer n, you have to make a pile of n levels of stones.
-- The first level has n stones.
-- The number of stones in the next level is:
--     - the next odd number if n is odd.
--     - the next even number if n is even.
-- Return the number of stones in each level in a list, where element at index
-- i represents the number of stones in the level (i+1).
--
-- Examples:
-- >>> make_a_pile 3
-- [3,5,7]
make_a_pile :: Int -> [Int]
make_a_pile n =
  [ n +  2 * i
    | i <-  [0 ..  n - 1]
  ]

-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (make_a_pile 3 == [3,5,7])
    check (make_a_pile 4 == [4,6,8,10])
    check (make_a_pile 5 == [5,7,9,11,13])
    check (make_a_pile 6 == [6,8,10,12,14,16])
    check (make_a_pile 8 == [8,10,12,14,16,18,20,22])
