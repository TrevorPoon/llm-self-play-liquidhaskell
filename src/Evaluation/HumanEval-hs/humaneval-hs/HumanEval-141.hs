-- Task ID: HumanEval/141
-- Assigned To: Author B

-- Python Implementation:

--
-- def file_name_check(file_name):
--     """Create a function which takes a string representing a file's name, and returns
--     'Yes' if the the file's name is valid, and returns 'No' otherwise.
--     A file's name is considered to be valid if and only if all the following conditions
--     are met:
--     - There should not be more than three digits ('0'-'9') in the file's name.
--     - The file's name contains exactly one dot '.'
--     - The substring before the dot should not be empty, and it starts with a letter from
--     the latin alphapet ('a'-'z' and 'A'-'Z').
--     - The substring after the dot should be one of these: ['txt', 'exe', 'dll']
--     Examples:
--     file_name_check("example.txt") # => 'Yes'
--     file_name_check("1example.dll") # => 'No' (the name should start with a latin alphapet letter)
--     """
--     suf = ['txt', 'exe', 'dll']
--     lst = file_name.split(sep='.')
--     if len(lst) != 2:
--         return 'No'
--     if not lst[1] in suf:
--         return 'No'
--     if len(lst[0]) == 0:
--         return 'No'
--     if not lst[0][0].isalpha():
--         return 'No'
--     t = len([x for x in lst[0] if x.isdigit()])
--     if t > 3:
--         return 'No'
--     return 'Yes'
--

-- Haskell Implementation:

-- Create a function which takes a string representing a file's name, and returns
-- 'Yes' if the the file's name is valid, and returns 'No' otherwise.
-- A file's name is considered to be valid if and only if all the following conditions are met:
-- - There should not be more than three digits ('0'-'9') in the file's name.
-- - The file's name contains exactly one dot '.'
-- - The substring before the dot should not be empty, and it starts with a letter from the latin alphapet ('a'-'z' and 'A'-'Z').
-- - The substring after the dot should be one of these: ['txt', 'exe', 'dll']
-- Examples:
-- >>> file_name_check "example.txt"
-- "Yes"
-- >>> file_name_check "1example.dll"
-- "No" (the name should start with a latin alphapet letter
file_name_check :: String -> String
file_name_check filename
  | (length filename) < 5 = "No"
  | otherwise =  if containsOneDot filename && containsNoMoreThanThreeDigits filename && endsOnTxtExeOrDll filename && doesNotStartWithDot filename && startWithLetterFromLatinAlphabet filename then  "Yes" else "No"
  where
    containsOneDot :: String -> Bool
    containsOneDot filename =  length (filter (== '.') filename) == 1
    containsNoMoreThanThreeDigits filename =  length (filter (\x -> x `elem` ['0' .. '9']) filename) <= 3
    endsOnTxtExeOrDll filename =  (drop (length filename - 4) filename) `elem` [".txt", ".exe", ".dll"]
    doesNotStartWithDot filename =  head filename /= '.'
    startWithLetterFromLatinAlphabet filename =  head filename `elem` ['a' .. 'z'] ||  head filename `elem` ['A' .. 'Z']
