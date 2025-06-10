-- Task ID: HumanEval/51
-- Assigned To: Author C

-- Python Implementation:

-- 
-- 
-- def remove_vowels(text):
--     """
--     remove_vowels is a function that takes string and returns string without vowels.
--     >>> remove_vowels('')
--     ''
--     >>> remove_vowels("abcdef\nghijklm")
--     'bcdf\nghjklm'
--     >>> remove_vowels('abcdef')
--     'bcdf'
--     >>> remove_vowels('aaaaa')
--     ''
--     >>> remove_vowels('aaBAA')
--     'B'
--     >>> remove_vowels('zbcd')
--     'zbcd'
--     """
--     return "".join([s for s in text if s.lower() not in ["a", "e", "i", "o", "u"]])
-- 


-- Haskell Implementation:

-- remove_vowels is a function that takes string and returns string without vowels.
-- >>> remove_vowels ""
-- ""
-- >>> remove_vowels "abcdef\nghijklm"
-- "bcdf\nghjklm"
-- >>> remove_vowels "abcdef"
-- "bcdf"
-- >>> remove_vowels "aaaaa"
-- ""
-- >>> remove_vowels "aaBAA"
-- "B"
-- >>> remove_vowels "zbcd"
-- "zbcd"
remove_vowels :: String -> String
remove_vowels =  filter (`notElem` "aeiouAEIOU")



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (remove_vowels "" == "")
    check (remove_vowels "abcdef\nghijklm" == "bcdf\nghjklm")
    check (remove_vowels "abcdef" == "bcdf")
    check (remove_vowels "aaaaa" == "")
    check (remove_vowels "aaBAA" == "B")
    check (remove_vowels "zbcd" == "zbcd")