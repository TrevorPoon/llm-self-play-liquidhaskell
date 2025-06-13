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
import Control.Monad (forM_, replicateM)
import System.Random (randomRIO)

-- Add two numbers x and y
-- >>> add 2 3
-- 5
-- >>> add 5 7
-- 12
add :: Int -> Int -> Int
add =  (+)
