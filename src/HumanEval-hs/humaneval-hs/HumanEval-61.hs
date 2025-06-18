-- Task ID: HumanEval/61
-- Assigned To: Author C

-- Python Implementation:

-- 
-- 
-- def correct_bracketing(brackets: str):
--     """ brackets is a string of "(" and ")".
--     return True if every opening bracket has a corresponding closing bracket.
-- 
--     >>> correct_bracketing("(")
--     False
--     >>> correct_bracketing("()")
--     True
--     >>> correct_bracketing("(()())")
--     True
--     >>> correct_bracketing(")(()")
--     False
--     """
--     depth = 0
--     for b in brackets:
--         if b == "(":
--             depth += 1
--         else:
--             depth -= 1
--         if depth < 0:
--             return False
--     return depth == 0
-- 


-- Haskell Implementation:

-- brackets is a string of "(" and ")".
-- return True if every opening bracket has a corresponding closing bracket.

-- >>> correct_bracketing "("
-- False
-- >>> correct_bracketing "()"
-- True
-- >>> correct_bracketing "(()())"
-- True
-- >>> correct_bracketing ")(()"
-- False
correct_bracketing :: String -> Bool
correct_bracketing = go 0
  where
    go :: Int -> String -> Bool
    go depth []        = depth == 0
    go depth (_:_) | depth < 0 = False
    go depth (c:cs)
      | c == '('        = go (depth + 1) cs
      | c == ')'        = go (depth - 1) cs
      | otherwise       = go depth       cs
