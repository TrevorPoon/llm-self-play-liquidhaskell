-- Task ID: HumanEval/91
-- Assigned To: Author E

-- Python Implementation:

--
-- def is_bored(S):
--     """
--     You'll be given a string of words, and your task is to count the number
--     of boredoms. A boredom is a sentence that starts with the word "I".
--     Sentences are delimited by '.', '?' or '!'.
--
--     For example:
--     >>> is_bored("Hello world")
--     0
--     >>> is_bored("The sky is blue. The sun is shining. I love this weather")
--     1
--     """
--     import re
--     sentences = re.split(r'[.?!]\s*', S)
--     return sum(sentence[0:2] == 'I ' for sentence in sentences)
--

-- Haskell Implementation:
import Data.List

-- You'll be given a string of words, and your task is to count the number
-- of boredoms. A boredom is a sentence that starts with the word "I".
-- Sentences are delimited by '.', '?' or '!'.

-- For example:
-- >>> is_bored "Hello world"
-- 0
-- >>> is_bored "The sky is blue. The sun is shining. I love this weather"
-- 1
is_bored :: String -> Int
is_bored s =
  sum
    [ 1
      | sentence <-  splitOnDelimiters s,
        take 2 sentence ==  "I "
    ]

-- Helper function to split the string based on delimiters '.', '!', and '?'
splitOnDelimiters :: String -> [String]
splitOnDelimiters [] =  []
splitOnDelimiters s = case break  (`elem` ".?!") s of
  (sentence, []) ->  [sentence]
  (sentence, (_ : rest)) -> sentence :  splitOnDelimiters (dropWhile  (`elem` " ") rest)

-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (is_bored "Hello world" == 0)
    check (is_bored "Is the sky blue?" == 0)
    check (is_bored "I love It !" == 1)
    check (is_bored "bIt" == 0)
    check (is_bored "I feel good today. I will be productive. will kill It" == 2)
    check (is_bored "You and I are going for a walk" == 0)
