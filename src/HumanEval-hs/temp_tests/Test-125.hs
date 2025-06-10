-- Task ID: HumanEval/125
-- Assigned To: Author D

-- Python Implementation:

-- 
-- def split_words(txt):
--     '''
--     Given a string of words, return a list of words split on whitespace, if no whitespaces exists in the text you
--     should split on commas ',' if no commas exists you should return the number of lower-case letters with odd order in the
--     alphabet, ord('a') = 0, ord('b') = 1, ... ord('z') = 25
--     Examples
--     split_words("Hello world!") ➞ ["Hello", "world!"]
--     split_words("Hello,world!") ➞ ["Hello", "world!"]
--     split_words("abcdef") == 3 
--     '''
--     if " " in txt:
--         return txt.split()
--     elif "," in txt:
--         return txt.replace(',',' ').split()
--     else:
--         return len([i for i in txt if i.islower() and ord(i)%2 == 0])
-- 


-- Haskell Implementation:
import Data.Char

-- Given a string of words, return a list of words split on whitespace, if no whitespaces exists in the text you
-- should split on commas ',' if no commas exists you should return the number of lower-case letters with odd order in the
-- alphabet, ord('a') = 0, ord('b') = 1, ... ord('z') = 25
-- Examples
-- split_words "Hello world!" ➞ ["Hello","world!"]
-- split_words "Hello,world!" ➞ ["Hello","world!"]
-- split_words "abcdef" == 3 

split_words :: String -> Either Int [String]
split_words txt
  | ' ' `elem` txt =  Right $ words txt
  | ',' `elem` txt =  Right $ words $  map (\c -> if c == ',' then ' ' else c) txt
  | otherwise =  Left $ length $ filter  (\c -> isLower c && even (ord c)) txt



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (split_words "Hello world!" == Right ["Hello","world!"])
    check (split_words "Hello,world!" == Right ["Hello","world!"])
    check (split_words "Hello world,!" == Right ["Hello","world,!"])
    check (split_words "Hello,Hello,world !" == Right ["Hello,Hello,world","!"])
    check (split_words "abcdef" == Left 3)
    check (split_words "aaabb" == Left 2)
    check (split_words "aaaBb" == Left 1)
    check (split_words "" == Left 0)
