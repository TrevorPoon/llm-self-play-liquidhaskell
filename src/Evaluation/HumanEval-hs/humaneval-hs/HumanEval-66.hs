-- Task ID: HumanEval/66
-- Assigned To: Author E

-- Python Implementation:

-- 
-- def digitSum(s):
--     """Task
--     Write a function that takes a string as input and returns the sum of the upper characters only'
--     ASCII codes.
-- 
--     Examples:
--         digitSum("") => 0
--         digitSum("abAB") => 131
--         digitSum("abcCd") => 67
--         digitSum("helloE") => 69
--         digitSum("woArBld") => 131
--         digitSum("aAaaaXa") => 153
--     """
--     if s == "": return 0
--     return sum(ord(char) if char.isupper() else 0 for char in s)
-- 


-- Haskell Implementation:
import Data.Char

-- Task
-- Write a function that takes a string as input and returns the sum of the upper characters only'
-- ASCII codes.
--
-- Examples:
--     digitSum "" => 0
--     digitSum "abAB" => 131
--     digitSum "abcCd" => 67
--     digitSum "helloE" => 69
--     digitSum "woArBld" => 131
--     digitSum "aAaaaXa" => 153
digitSum :: String -> Int
digitSum s =  sum [ord c | c <- s,  isUpper c]
