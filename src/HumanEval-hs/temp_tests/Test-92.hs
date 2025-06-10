-- Task ID: HumanEval/92
-- Assigned To: Author E

-- Python Implementation:

--
-- def any_int(x, y, z):
--     '''
--     Create a function that takes 3 numbers.
--     Returns true if one of the numbers is equal to the sum of the other two, and all numbers are integers.
--     Returns false in any other cases.
--
--     Examples
--     any_int(5, 2, 7) ➞ True
--
--     any_int(3, 2, 2) ➞ False
--
--     any_int(3, -2, 1) ➞ True
--
--     any_int(3.6, -2.2, 2) ➞ False
--
--
--
--     '''
--
--     if isinstance(x,int) and isinstance(y,int) and isinstance(z,int):
--         if (x+y==z) or (x+z==y) or (y+z==x):
--             return True
--         return False
--     return False
--

-- Haskell Implementation:
import Data.Typeable

-- Create a function that takes 3 numbers.
-- Returns true if one of the numbers is equal to the sum of the other two, and all numbers are integers.
-- Returns false in any other cases.
--
-- Examples
-- any_int 5 2 7 => True
--
-- any_int 3 2 2 => False
--
-- any_int 3 (-2) 1 => True
--
-- any_int 3.6 (-2.2) 2 => False

any_int :: Float -> Float -> Float -> Bool
any_int x y z =
  isInt x
    && isInt y
    && isInt  z
    && (x + y == z ||  x + z == y ||  y + z == x)
  where
    isInt a = a ==  fromInteger (round a)

-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (any_int 2 3 1 == True)
    check (any_int 2.5 2 3 == False)
    check (any_int 1.5 5 3.5 == False)
    check (any_int 2 6 2 == False)
    check (any_int 4 2 2 == True)
    check (any_int 2.2 2.2 2.2 == False)
    check (any_int (-4) 6 2 == True)
    check (any_int 2 1 1 == True)
    check (any_int 3 4 7 == True)
    check (any_int 3.0 4 7 == False)
