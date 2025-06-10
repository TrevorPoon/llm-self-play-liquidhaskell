-- Task ID: HumanEval/89
-- Assigned To: Author E

-- Python Implementation:

--
-- def encrypt(s):
--     """Create a function encrypt that takes a string as an argument and
--     returns a string encrypted with the alphabet being rotated.
--     The alphabet should be rotated in a manner such that the letters
--     shift down by two multiplied to two places.
--     For example:
--     encrypt('hi') returns 'lm'
--     encrypt('asdfghjkl') returns 'ewhjklnop'
--     encrypt('gf') returns 'kj'
--     encrypt('et') returns 'ix'
--     """
--     d = 'abcdefghijklmnopqrstuvwxyz'
--     out = ''
--     for c in s:
--         if c in d:
--             out += d[(d.index(c)+2*2) % 26]
--         else:
--             out += c
--     return out
--

-- Haskell Implementation:
import Data.Char

-- Create a function encrypt that takes a string as an argument and
-- returns a string encrypted with the alphabet being rotated.
-- The alphabet should be rotated in a manner such that the letters
-- shift down by two multiplied to two places.
-- For example:
-- >>> encrypt "hi"
-- "lm"
-- >>> encrypt "asdfghjkl"
-- "ewhjklnop"
-- >>> encrypt "gf"
-- "kj"
-- >>> encrypt "et"
-- "ix"

encrypt :: String -> String
encrypt s =
  let d =  ['a' .. 'z']
   in [ d !!  ((fromEnum c - fromEnum 'a' +  2 * 2) `mod`  26)
        | c <-  s
      ]



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (encrypt "hi"                                            == "lm")
    check (encrypt "asdfghjkl"                                     == "ewhjklnop")
    check (encrypt "gf"                                            == "kj")
    check (encrypt "et"                                            == "ix")
    check (encrypt "faewfawefaewg"                                == "jeiajeaijeiak")
    check (encrypt "hellomyfriend"                                == "lippsqcjvmirh")
    check (encrypt "dxzdlmnilfuhmilufhlihufnmlimnufhlimnufhfucufh" == "hbdhpqrmpjylqmpyjlpmlyjrqpmqryjlpmqryjljygyjl")
    check (encrypt "a"                                             == "e")
