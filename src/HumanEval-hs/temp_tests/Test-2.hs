-- Task ID: HumanEval/2
-- Assigned To: Author A

-- Python Implementation:

-- 
-- 
-- def truncate_number(number: float) -> float:
--     """ Given a positive floating point number, it can be decomposed into
--     and integer part (largest integer smaller than given number) and decimals
--     (leftover part always smaller than 1).
-- 
--     Return the decimal part of the number.
--     >>> truncate_number(3.5)
--     0.5
--     """
--     return number % 1.0
-- 


-- Haskell Implementation:

-- Given a positive floating point number, it can be decomposed into
-- and integer part (largest integer smaller than given number) and decimals
-- (leftover part always smaller than 1).
--
-- Return the decimal part of the number.
-- >>> truncate_number 3.5
-- 0.5
truncate_number :: Float -> Float
truncate_number number =  number - fromIntegral  (floor number)



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (truncate_number 3.5 == 0.5)
    check (abs (truncate_number 1.33   - 0.33 ) < 1e-6)
    check (abs (truncate_number 123.456 - 0.456) < 1e-6)
