-- Task ID: HumanEval/16
-- Assigned To: Author A

-- Python Implementation:

-- 
-- 
-- def count_distinct_characters(string: str) -> int:
--     """ Given a string, find out how many distinct characters (regardless of case) does it consist of
--     >>> count_distinct_characters('xyzXYZ')
--     3
--     >>> count_distinct_characters('Jerry')
--     4
--     """
--     return len(set(string.lower()))
-- 


-- Haskell Implementation:
import Data.Char
import Data.List

-- Given a string, find out how many distinct characters (regardless of case) does it consist of
-- >>> count_distinct_characters "xyzXYZ"
-- 3
-- >>> count_distinct_characters "Jerry"
-- 4
count_distinct_characters :: String -> Int
count_distinct_characters string =  length $  nub $  map toLower string



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (count_distinct_characters ""                  == 0)
    check (count_distinct_characters "abcde"             == 5)
    check (count_distinct_characters ("abcde" ++ "cade" ++ "CADE") == 5)
    check (count_distinct_characters "aaaaAAAAaaaa"      == 1)
    check (count_distinct_characters "Jerry jERRY JeRRRY" == 5)
