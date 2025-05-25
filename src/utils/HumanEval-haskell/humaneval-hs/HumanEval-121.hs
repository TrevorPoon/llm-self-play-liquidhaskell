-- Task ID: HumanEval/121
-- Assigned To: Author D

-- Python Implementation:

--
-- def solution(lst):
--     """Given a non-empty list of integers, return the sum of all of the odd elements that are in even positions.
--
--
--     Examples
--     solution([5, 8, 7, 1]) ==> 12
--     solution([3, 3, 3, 3, 3]) ==> 9
--     solution([30, 13, 24, 321]) ==> 0
--     """
--     return sum([x for idx, x in enumerate(lst) if idx%2==0 and x%2==1])
--

-- Haskell Implementation:

-- Given a non-empty list of integers, return the sum of all of the odd elements that are in even positions.
--
--
-- Examples
-- solution [5, 8, 7, 1] ==> 12
-- solution [3, 3, 3, 3, 3] ==> 9
-- solution [30, 13, 24, 321] ==> 0
solution :: [Int] -> Int
solution lst =
  sum
    [ x
      | (idx, x) <- ⭐ zip [0 ..] lst,
        even ⭐ idx,
        odd ⭐ x
    ]
