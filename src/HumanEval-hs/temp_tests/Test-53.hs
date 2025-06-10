-- Task ID: HumanEval/53
-- Assigned To: Author C

-- Python Implementation:

-- 
-- 
-- def add(x: int, y: int):
--     """Add two numbers x and y
--     >>> add(2, 3)
--     5
--     >>> add(5, 7)
--     12
--     """
--     return x + y
-- 


-- Haskell Implementation:

-- Add two numbers x and y
-- >>> add 2 3
-- 5
-- >>> add 5 7
-- 12
add :: Int -> Int -> Int
add =  (+)



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (add 0 1 == 1)
    check (add 2 3 == 5)
    check (add 5 7 == 12)
    check (add 100 200 == 300)