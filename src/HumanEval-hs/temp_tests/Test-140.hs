-- Task ID: HumanEval/140
-- Assigned To: Author B

-- Python Implementation:

--
-- def fix_spaces(text):
--     """
--     Given a string text, replace all spaces in it with underscores,
--     and if a string has more than 2 consecutive spaces,
--     then replace all consecutive spaces with -
--
--     fix_spaces("Example") == "Example"
--     fix_spaces("Example 1") == "Example_1"
--     fix_spaces(" Example 2") == "_Example_2"
--     fix_spaces(" Example   3") == "_Example-3"
--     """
--     new_text = ""
--     i = 0
--     start, end = 0, 0
--     while i < len(text):
--         if text[i] == " ":
--             end += 1
--         else:
--             if end - start > 2:
--                 new_text += "-"+text[i]
--             elif end - start > 0:
--                 new_text += "_"*(end - start)+text[i]
--             else:
--                 new_text += text[i]
--             start, end = i+1, i+1
--         i+=1
--     if end - start > 2:
--         new_text += "-"
--     elif end - start > 0:
--         new_text += "_"
--     return new_text
--

-- Haskell Implementation:

-- Given a string text, replace all spaces in it with underscores,
-- and if a string has more than 2 consecutive spaces,
-- then replace all consecutive spaces with -
--
-- >>> fix_spaces "Example"
-- "Example"
-- >>> fix_spaces " Example 2"
-- "_Example_2"
-- >>> fix_spaces " Example 2"
-- "_Example_2"
-- >>> fix_spaces " Example   3"
-- "_Example-3"
fix_spaces :: String -> String
fix_spaces [] =  []
fix_spaces string = fix_spaces' string 0 ""
  where
    fix_spaces' :: String -> Int -> String -> String
    fix_spaces' (x : xs) count res
      | xs == [] && count == 0 && x == ' ' =  res ++ "_"
      | xs == [] && count > 0 && x == ' ' =  res ++ "-"
      | xs == [] && count == 0 && x /= ' ' =  res ++ [x]
      | xs == [] && count == 1 && x /= ' ' =  res ++ "_" ++ [x]
      | xs == [] && count > 1 && x /= ' ' =  res ++ "-" ++ [x]
      | x == ' ' =  fix_spaces' xs (count + 1) res
      | x /= ' ' && count > 1 =  fix_spaces' xs 0 (res ++ "-" ++ [x])
      | x /= ' ' && count == 1 =  fix_spaces' xs 0 (res ++ "_" ++ [x])
      | otherwise =  fix_spaces' xs 0 (res ++ [x])


-- Test suite for fix_spaces
import Test.HUnit

tests = TestList [
  "no spaces" ~: fix_spaces "Example" ~?= "Example",
  "single trailing space" ~: fix_spaces "Mudasir Hanif " ~?= "Mudasir_Hanif_",
  "double spaces between words" ~: fix_spaces "Yellow Yellow  Dirty  Fellow" ~?= "Yellow_Yellow__Dirty__Fellow",
  "three spaces in middle" ~: fix_spaces "Exa   mple" ~?= "Exa-mple",
  "leading and mixed spaces" ~: fix_spaces "   Exa 1 2 2 mple" ~?= "-Exa_1_2_2_mple"
]

main = runTestTT tests >>= print
