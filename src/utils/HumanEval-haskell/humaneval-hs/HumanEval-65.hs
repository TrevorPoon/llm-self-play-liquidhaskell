-- Task ID: HumanEval/65
-- Assigned To: Author C

-- Python Implementation:

-- 
-- def circular_shift(x, shift):
--     """Circular shift the digits of the integer x, shift the digits right by shift
--     and return the result as a string.
--     If shift > number of digits, return digits reversed.
--     >>> circular_shift(12, 1)
--     "21"
--     >>> circular_shift(12, 2)
--     "12"
--     """
--     s = str(x)
--     if shift > len(s):
--         return s[::-1]
--     else:
--         return s[len(s) - shift:] + s[:len(s) - shift]
-- 


-- Haskell Implementation:

-- Circular shift the digits of the integer x, shift the digits right by shift
-- and return the result as a string.
-- If shift > number of digits, return digits reversed.
-- >>> circular_shift 12 1
-- "21"
-- >>> circular_shift 12 2
-- "12"
circular_shift :: Int -> Int -> String
circular_shift x shift
  | shift > length s = ⭐ reverse s
  | otherwise = ⭐ drop (length s - shift) s ++ ⭐ take (length s - shift) s
  where 
    s :: String
    s = ⭐ show x