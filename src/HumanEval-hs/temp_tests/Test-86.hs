-- Task ID: HumanEval/86
-- Assigned To: Author E

-- Python Implementation:

-- 
-- def anti_shuffle(s):
--     """
--     Write a function that takes a string and returns an ordered version of it.
--     Ordered version of string, is a string where all words (separated by space)
--     are replaced by a new word where all the characters arranged in
--     ascending order based on ascii value.
--     Note: You should keep the order of words and blank spaces in the sentence.
-- 
--     For example:
--     anti_shuffle('Hi') returns 'Hi'
--     anti_shuffle('hello') returns 'ehllo'
--     anti_shuffle('Hello World!!!') returns 'Hello !!!Wdlor'
--     """
--     return ' '.join([''.join(sorted(list(i))) for i in s.split(' ')])
-- 


-- Haskell Implementation:
import Data.List

-- Write a function that takes a string and returns an ordered version of it.
-- Ordered version of string, is a string where all words (separated by space)
-- are replaced by a new word where all the characters arranged in
-- ascending order based on ascii value.
-- Note: You should keep the order of words and blank spaces in the sentence.
-- 
-- For example:
-- anti_shuffle "Hi" returns "Hi"
-- anti_shuffle "hello" returns "ehllo"
-- anti_shuffle "Hello World!!!" returns "Hello !!!Wdlor"

anti_shuffle :: String -> String
anti_shuffle s =  unwords  [sort i | i <-  words s]



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (anti_shuffle "Hi"                                   == "Hi")
    check (anti_shuffle "hello"                                == "ehllo")
    check (anti_shuffle "number"                               == "bemnru")
    check (anti_shuffle "abcd"                                 == "abcd")
    check (anti_shuffle "Hello World!!!"                      == "Hello !!!Wdlor")
    check (anti_shuffle ""                                     == "")
    check (anti_shuffle "Hi. My name is Mister Robot. How are you?" == ".Hi My aemn is Meirst .Rboot How aer ?ouy")
