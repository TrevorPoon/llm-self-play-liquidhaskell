-- Task ID: HumanEval/85
-- Assigned To: Author E

-- Python Implementation:

--
-- def add(lst):
--     """Given a non-empty list of integers lst. add the even elements that are at odd indices..
--
--
--     Examples:
--         add([4, 2, 6, 7]) ==> 2
--     """
--     return sum([lst[i] for i in range(1, len(lst), 2) if lst[i]%2 == 0])
--

-- Haskell Implementation:

-- Given a non-empty list of integers lst. add the even elements that are at odd indices..
--
-- Examples:
--     add [4, 2, 6, 7] ==> 2

add :: [Int] -> Int
add lst =
  sum
    [ lst !! i
      | i <-  [1, 3 ..  (length lst - 1)],
        even  (lst !! i)
    ]



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (add [4,88]        == 88)
    check (add [4,5,6,7,2,122]== 122)
    check (add [4,0,6,7]     == 0)
    check (add [4,4,6,8]     == 12)
