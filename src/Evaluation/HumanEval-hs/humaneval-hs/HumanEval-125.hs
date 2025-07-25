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
