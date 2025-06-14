-- Task ID: HumanEval/149
-- Assigned To: Author B

-- Python Implementation:

--
-- def sorted_list_sum(lst):
--     """Write a function that accepts a list of strings as a parameter,
--     deletes the strings that have odd lengths from it,
--     and returns the resulted list with a sorted order,
--     The list is always a list of strings and never an array of numbers,
--     and it may contain duplicates.
--     The order of the list should be ascending by length of each word, and you
--     should return the list sorted by that rule.
--     If two words have the same length, sort the list alphabetically.
--     The function should return a list of strings in sorted order.
--     You may assume that all words will have the same length.
--     For example:
--     assert list_sort(["aa", "a", "aaa"]) => ["aa"]
--     assert list_sort(["ab", "a", "aaa", "cd"]) => ["ab", "cd"]
--     """
--     lst.sort()
--     new_lst = []
--     for i in lst:
--         if len(i)%2 == 0:
--             new_lst.append(i)
--     return sorted(new_lst, key=len)
--

-- Haskell Implementation:

-- Write a function that accepts a list of strings as a parameter,
-- deletes the strings that have odd lengths from it,
-- and returns the resulted list with a sorted order,
-- The list is always a list of strings and never an array of numbers,
-- and it may contain duplicates.
-- The order of the list should be ascending by length of each word, and you
-- should return the list sorted by that rule.
-- If two words have the same length, sort the list alphabetically.
-- The function should return a list of strings in sorted order.
-- You may assume that all words will have the same length.
-- For example:
-- >>> sorted_list_sum ["aa", "a", "aaa"]
-- ["aa"]
-- >>> sorted_list_sum ["ab", "a", "aaa", "cd"]
-- ["ab", "cd"]
import Data.List (sort)

sorted_list_sum :: [String] -> [String]
sorted_list_sum lst =  sort $ filter  (\x -> length x `mod` 2 == 0) lst
