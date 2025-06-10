-- Task ID: HumanEval/11
-- Assigned To: Author A

-- Python Implementation:

-- from typing import List
-- 
-- 
-- def string_xor(a: str, b: str) -> str:
--     """ Input are two strings a and b consisting only of 1s and 0s.
--     Perform binary XOR on these inputs and return result also as a string.
--     >>> string_xor('010', '110')
--     '100'
--     """
--     def xor(i, j):
--         if i == j:
--             return '0'
--         else:
--             return '1'
-- 
--     return ''.join(xor(x, y) for x, y in zip(a, b))
-- 


-- Haskell Implementation:

-- Input are two strings a and b consisting only of 1s and 0s.
-- Perform binary XOR on these inputs and return result also as a string.
-- >>> string_xor "010" "110"
-- "100"
string_xor :: String -> String -> String
string_xor a b =  [if x == y  then '0' else '1' |  (x, y) <-  zip a b]



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (string_xor "111000" "101010" == "010010")
    check (string_xor "1" "1"         == "0")
    check (string_xor "0101" "0000"   == "0101")
