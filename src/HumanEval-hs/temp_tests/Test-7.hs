-- Task ID: HumanEval/7
-- Assigned To: Author A

-- Python Implementation:

-- from typing import List
-- 
-- 
-- def filter_by_substring(strings: List[str], substring: str) -> List[str]:
--     """ Filter an input list of strings only for ones that contain given substring
--     >>> filter_by_substring([], 'a')
--     []
--     >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')
--     ['abc', 'bacd', 'array']
--     """
--     return [x for x in strings if substring in x]
-- 


-- Haskell Implementation:
import Data.List

-- Filter an input list of strings only for ones that contain given substring
-- >>> filter_by_substring [] "a"
-- []
-- >>> filter_by_substring ["abc", "bacd", "cde", "array"] "a"
-- ["abc","bacd","array"]
filter_by_substring :: [String] -> String -> [String]
filter_by_substring strings substring =  [x | x <- strings,  substring `isInfixOf` x]



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (filter_by_substring []                          "john" == [])
    check (filter_by_substring ["xxx","asd","xxy","john doe","xxxAAA","xxx"] "xxx" == ["xxx","xxxAAA","xxx"])
    check (filter_by_substring ["xxx","asd","aaaxxy","john doe","xxxAAA","xxx"] "xx"  == ["xxx","aaaxxy","xxxAAA","xxx"])
    check (filter_by_substring ["grunt","trumpet","prune","gruesome"]    "run" == ["grunt","prune"])
