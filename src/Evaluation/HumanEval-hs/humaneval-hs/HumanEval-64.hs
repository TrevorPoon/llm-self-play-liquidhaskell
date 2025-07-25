-- Task ID: HumanEval/64
-- Assigned To: Author C

-- Python Implementation:

-- 
-- FIX = """
-- Add more test cases.
-- """
-- 
-- def vowels_count(s):
--     """Write a function vowels_count which takes a string representing
--     a word as input and returns the number of vowels in the string.
--     Vowels in this case are 'a', 'e', 'i', 'o', 'u'. Here, 'y' is also a
--     vowel, but only when it is at the end of the given word.
-- 
--     Example:
--     >>> vowels_count("abcde")
--     2
--     >>> vowels_count("ACEDY")
--     3
--     """
--     vowels = "aeiouAEIOU"
--     n_vowels = sum(c in vowels for c in s)
--     if s[-1] == 'y' or s[-1] == 'Y':
--         n_vowels += 1
--     return n_vowels
-- 


-- Haskell Implementation:

-- Write a function vowels_count which takes a string representing
-- a word as input and returns the number of vowels in the string.
-- Vowels in this case are 'a', 'e', 'i', 'o', 'u'. Here, 'y' is also a
-- vowel, but only when it is at the end of the given word.

-- Example:
-- >>> vowels_count "abcde"
-- 2
-- >>> vowels_count "ACEDY"
-- 3
vowels_count :: String -> Int
vowels_count s =  (length $ filter (`elem` "aeiouAEIOU") s) + if last s `elem` "yY" then  1 else 0
