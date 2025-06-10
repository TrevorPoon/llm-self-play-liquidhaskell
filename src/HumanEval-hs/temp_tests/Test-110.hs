-- Task ID: HumanEval/110
-- Assigned To: Author D

-- Python Implementation:

-- 
-- def exchange(lst1, lst2):
--     """In this problem, you will implement a function that takes two lists of numbers,
--     and determines whether it is possible to perform an exchange of elements
--     between them to make lst1 a list of only even numbers.
--     There is no limit on the number of exchanged elements between lst1 and lst2.
--     If it is possible to exchange elements between the lst1 and lst2 to make
--     all the elements of lst1 to be even, return "YES".
--     Otherwise, return "NO".
--     For example:
--     exchange([1, 2, 3, 4], [1, 2, 3, 4]) => "YES"
--     exchange([1, 2, 3, 4], [1, 5, 3, 4]) => "NO"
--     It is assumed that the input lists will be non-empty.
--     """
--     odd = 0
--     even = 0
--     for i in lst1:
--         if i%2 == 1:
--             odd += 1
--     for i in lst2:
--         if i%2 == 0:
--             even += 1
--     if even >= odd:
--         return "YES"
--     return "NO"
--             
-- 


-- Haskell Implementation:

-- In this problem, you will implement a function that takes two lists of numbers,
-- and determines whether it is possible to perform an exchange of elements
-- between them to make lst1 a list of only even numbers.
-- There is no limit on the number of exchanged elements between lst1 and lst2.
-- If it is possible to exchange elements between the lst1 and lst2 to make
-- all the elements of lst1 to be even, return "YES".
-- Otherwise, return "NO".
-- For example:
-- exchange [1, 2, 3, 4] [1, 2, 3, 4] => "YES"
-- exchange [1, 2, 3, 4] [1, 5, 3, 4] => "NO"
-- It is assumed that the input lists will be non-empty.
exchange :: [Int] -> [Int] -> String
exchange lst1 lst2
  | evenCount  >= oddCount =  "YES"
  | otherwise =  "NO"
  where
    evenCount =  length $  filter even  lst2
    oddCount =  length $  filter odd  lst1



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (exchange [1,2,3,4] [1,2,3,4] == "YES")
    check (exchange [1,2,3,4] [1,5,3,4] == "NO")
    check (exchange [1,2,3,4] [2,1,4,3] == "YES")
    check (exchange [5,7,3]   [2,6,4]   == "YES")
    check (exchange [5,7,3]   [2,6,3]   == "NO")
    check (exchange [3,2,6,1,8,9] [3,5,5,1,1,1] == "NO")
    check (exchange [100,200] [200,200] == "YES")
