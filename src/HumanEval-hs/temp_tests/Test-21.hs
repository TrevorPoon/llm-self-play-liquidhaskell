-- Task ID: HumanEval/21
-- Assigned To: Author A

-- Python Implementation:

-- from typing import List
-- 
-- 
-- def rescale_to_unit(numbers: List[float]) -> List[float]:
--     """ Given list of numbers (of at least two elements), apply a linear transform to that list,
--     such that the smallest number will become 0 and the largest will become 1
--     >>> rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])
--     [0.0, 0.25, 0.5, 0.75, 1.0]
--     """
--     min_number = min(numbers)
--     max_number = max(numbers)
--     return [(x - min_number) / (max_number - min_number) for x in numbers]
-- 


-- Haskell Implementation:

-- Given list of numbers (of at least two elements), apply a linear transform to that list,
-- such that the smallest number will become 0 and the largest will become 1
-- >>> rescale_to_unit [1.0, 2.0, 3.0, 4.0, 5.0]
-- [0.0,0.25,0.5,0.75,1.0]
rescale_to_unit :: [Float] -> [Float]
rescale_to_unit numbers = map  (\x ->  (x - min_number) /  (max_number - min_number))  numbers
    where min_number =  minimum numbers
          max_number =  maximum numbers



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (rescale_to_unit [2.0,49.9] == [0.0,1.0])
    check (rescale_to_unit [100.0,49.9] == [1.0,0.0])
    check (rescale_to_unit [1.0,2.0,3.0,4.0,5.0] == [0.0,0.25,0.5,0.75,1.0])
    check (rescale_to_unit [2.0,1.0,5.0,3.0,4.0] == [0.25,0.0,1.0,0.5,0.75])
    check (rescale_to_unit [12.0,11.0,15.0,13.0,14.0] == [0.25,0.0,1.0,0.5,0.75])
