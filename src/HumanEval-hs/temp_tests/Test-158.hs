-- Task ID: HumanEval/158
-- Assigned To: Author B

-- Python Implementation:

--
-- def find_max(words):
--     """Write a function that accepts a list of strings.
--     The list contains different words. Return the word with maximum number
--     of unique characters. If multiple strings have maximum number of unique
--     characters, return the one which comes first in lexicographical order.
--
--     find_max(["name", "of", "string"]) == "string"
--     find_max(["name", "enam", "game"]) == "enam"
--     find_max(["aaaaaaa", "bb" ,"cc"]) == ""aaaaaaa"
--     """
--     return sorted(words, key = lambda x: (-len(set(x)), x))[0]
--

-- Haskell Implementation:

-- Write a function that accepts a list of strings.
-- The list contains different words. Return the word with maximum number
-- of unique characters. If multiple strings have maximum number of unique
-- characters, return the one which comes first in lexicographical order.
--
-- >>> find_max ["name", "of", "string"]
-- "string"
-- >>> find_max ["name", "enam", "game"]
-- "enam"
-- >>> find_max ["aaaaaaa", "bb" ,"cc"]
-- "aaaaaaa"
import Data.List (nub, sort)

find_max :: [String] -> String
find_max words
  | words == [] =  ""
  | otherwise =  head $ filter (\x ->  (length . nub) x == maximum (map  (length . nub) words)) $  sort words



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (find_max ["name","of","string"] == "string")
    check (find_max ["name","enam","game"] == "enam")
    check (find_max ["aaaaaaa","bb","cc"] == "aaaaaaa")
    check (find_max ["abc","cba"] == "abc")
    check (find_max ["play","this","game","of","footbott"] == "footbott")
    check (find_max ["we","are","gonna","rock"] == "gonna")
    check (find_max ["we","are","a","mad","nation"] == "nation")
    check (find_max ["this","is","a","prrk"] == "this")
    check (find_max ["b"] == "b")
    check (find_max ["play","play","play"] == "play")
