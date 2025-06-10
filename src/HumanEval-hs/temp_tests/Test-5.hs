-- Task ID: HumanEval/5
-- Assigned To: Author A

-- Python Implementation:

-- from typing import List
-- 
-- 
-- def intersperse(numbers: List[int], delimeter: int) -> List[int]:
--     """ Insert a number 'delimeter' between every two consecutive elements of input list `numbers'
--     >>> intersperse([], 4)
--     []
--     >>> intersperse([1, 2, 3], 4)
--     [1, 4, 2, 4, 3]
--     """
--     if not numbers:
--         return []
-- 
--     result = []
-- 
--     for n in numbers[:-1]:
--         result.append(n)
--         result.append(delimeter)
-- 
--     result.append(numbers[-1])
-- 
--     return result
-- 


-- Haskell Implementation:

-- Insert a number 'delimeter' between every two consecutive elements of input list `numbers'
-- >>> intersperse [] 4
-- []
-- >>> intersperse [1, 2, 3] 4
-- [1,4,2,4,3,4]
intersperse :: [Int] -> Int -> [Int]
intersperse numbers delimeter =  concat [[x, delimeter] |  x <- numbers]



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (intersperse []       7 == [])
    check (intersperse [5,6,3,2] 8 == [5,8,6,8,3,8,2])
    check (intersperse [2,2,2]   2 == [2,2,2,2,2])
