-- Task ID: HumanEval/18
-- Assigned To: Author A

-- Python Implementation:

-- 
-- 
-- def how_many_times(string: str, substring: str) -> int:
--     """ Find how many times a given substring can be found in the original string. Count overlaping cases.
--     >>> how_many_times('', 'a')
--     0
--     >>> how_many_times('aaa', 'a')
--     3
--     >>> how_many_times('aaaa', 'aa')
--     3
--     """
--     times = 0
-- 
--     for i in range(len(string) - len(substring) + 1):
--         if string[i:i+len(substring)] == substring:
--             times += 1
-- 
--     return times
-- 


-- Haskell Implementation:
import Data.List

-- Find how many times a given substring can be found in the original string. Count overlaping cases.
-- >>> how_many_times "" "a"
-- 0
-- >>> how_many_times "aaa" "a"
-- 3
-- >>> how_many_times "aaaa" "aa"
-- 3
how_many_times :: String -> String -> Int
how_many_times string substring =  length $ filter  (substring `isPrefixOf`) $ map  (take  (length substring)) $  tails string




-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (how_many_times ""          "x"    == 0)
    check (how_many_times "xyxyxyx"  "x"    == 4)
    check (how_many_times "cacacacac" "cac"  == 4)
    check (how_many_times "john doe"  "john" == 1)
