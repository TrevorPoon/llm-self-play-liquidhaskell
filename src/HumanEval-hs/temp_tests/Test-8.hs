-- Task ID: HumanEval/8
-- Assigned To: Author A

-- Python Implementation:

-- from typing import List, Tuple
-- 
-- 
-- def sum_product(numbers: List[int]) -> Tuple[int, int]:
--     """ For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.
--     Empty sum should be equal to 0 and empty product should be equal to 1.
--     >>> sum_product([])
--     (0, 1)
--     >>> sum_product([1, 2, 3, 4])
--     (10, 24)
--     """
--     sum_value = 0
--     prod_value = 1
-- 
--     for n in numbers:
--         sum_value += n
--         prod_value *= n
--     return sum_value, prod_value
-- 


-- Haskell Implementation:

-- For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.
-- Empty sum should be equal to 0 and empty product should be equal to 1.
-- >>> sum_product []
-- (0,1)
-- >>> sum_product [1, 2, 3, 4]
-- (10,24)
sum_product :: [Int] -> (Int, Int)
sum_product numbers =  (sum numbers, product numbers)



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (sum_product []       == (0,1))
    check (sum_product [1,1,1]  == (3,1))
    check (sum_product [100,0]  == (100,0))
    check (sum_product [3,5,7]  == (15,105))
    check (sum_product [10]     == (10,10))
