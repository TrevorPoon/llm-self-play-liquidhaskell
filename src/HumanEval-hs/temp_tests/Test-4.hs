-- Task ID: HumanEval/4
-- Assigned To: Author A

-- Python Implementation:

-- from typing import List
-- 
-- 
-- def mean_absolute_deviation(numbers: List[float]) -> float:
--     """ For a given list of input numbers, calculate Mean Absolute Deviation
--     around the mean of this dataset.
--     Mean Absolute Deviation is the average absolute difference between each
--     element and a centerpoint (mean in this case):
--     MAD = average | x - x_mean |
--     >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
--     1.0
--     """
--     mean = sum(numbers) / len(numbers)
--     return sum(abs(x - mean) for x in numbers) / len(numbers)
-- 


-- Haskell Implementation:

-- For a given list of input numbers, calculate Mean Absolute Deviation
-- around the mean of this dataset.
-- Mean Absolute Deviation is the average absolute difference between each
-- element and a centerpoint (mean in this case):
-- MAD = average | x - x_mean |
-- >>> mean_absolute_deviation [1.0, 2.0, 3.0, 4.0]
-- 1.0
mean_absolute_deviation :: [Float] -> Float
mean_absolute_deviation numbers = sum  (map abs (map (\x ->  x - mean) numbers)) /  fromIntegral (length numbers)
    where
        mean =  sum numbers /  fromIntegral (length numbers)



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (abs (mean_absolute_deviation [1.0,2.0,3.0]       - 2.0/3.0) < 1e-6)
    check (abs (mean_absolute_deviation [1.0,2.0,3.0,4.0]   - 1.0     ) < 1e-6)
    check (abs (mean_absolute_deviation [1.0,2.0,3.0,4.0,5.0] - 6.0/5.0) < 1e-6)
