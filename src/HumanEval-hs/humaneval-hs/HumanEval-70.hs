-- Task ID: HumanEval/70
-- Assigned To: Author E

-- Python Implementation:

-- 
-- def strange_sort_list(lst):
--     '''
--     Given list of integers, return list in strange order.
--     Strange sorting, is when you start with the minimum value,
--     then maximum of the remaining integers, then minimum and so on.
-- 
--     Examples:
--     strange_sort_list([1, 2, 3, 4]) == [1, 4, 2, 3]
--     strange_sort_list([5, 5, 5, 5]) == [5, 5, 5, 5]
--     strange_sort_list([]) == []
--     '''
--     res, switch = [], True
--     while lst:
--         res.append(min(lst) if switch else max(lst))
--         lst.remove(res[-1])
--         switch = not switch
--     return res
-- 


-- Haskell Implementation:
import Data.List

-- Given list of integers, return list in strange order.
-- Strange sorting, is when you start with the minimum value,
-- then maximum of the remaining integers, then minimum and so on.
--
-- Examples:
-- strange_sort_list [1, 2, 3, 4] == [1, 4, 2, 3]
-- strange_sort_list [5, 5, 5, 5] == [5, 5, 5, 5]
-- strange_sort_list [] == []

strange_sort_list :: [Int] -> [Int]
strange_sort_list lst = helper lst True
  where
    helper :: [Int] -> Bool -> [Int]
    helper [] _ = []
    helper xs True  =
      let m = minimum xs
      in  m : helper (delete m xs) False
    helper xs False =
      let m = maximum xs
      in  m : helper (delete m xs) True
                        