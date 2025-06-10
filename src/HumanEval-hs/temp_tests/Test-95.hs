-- Task ID: HumanEval/95
-- Assigned To: Author E

-- Python Implementation:

--
-- def check_dict_case(dict):
--     """
--     Given a dictionary, return True if all keys are strings in lower
--     case or all keys are strings in upper case, else return False.
--     The function should return False is the given dictionary is empty.
--     Examples:
--     check_dict_case({"a":"apple", "b":"banana"}) should return True.
--     check_dict_case({"a":"apple", "A":"banana", "B":"banana"}) should return False.
--     check_dict_case({"a":"apple", 8:"banana", "a":"apple"}) should return False.
--     check_dict_case({"Name":"John", "Age":"36", "City":"Houston"}) should return False.
--     check_dict_case({"STATE":"NC", "ZIP":"12345" }) should return True.
--     """
--     if len(dict.keys()) == 0:
--         return False
--     else:
--         state = "start"
--         for key in dict.keys():
--
--             if isinstance(key, str) == False:
--                 state = "mixed"
--                 break
--             if state == "start":
--                 if key.isupper():
--                     state = "upper"
--                 elif key.islower():
--                     state = "lower"
--                 else:
--                     break
--             elif (state == "upper" and not key.isupper()) or (state == "lower" and not key.islower()):
--                     state = "mixed"
--                     break
--             else:
--                 break
--         return state == "upper" or state == "lower"
--

-- Haskell Implementation:
import Data.Char

-- Given a dictionary, return True if all keys are strings in lower
-- case or all keys are strings in upper case, else return False.
-- The function should return False is the given dictionary is empty.
-- Examples:
-- check_dict_case [("a","apple"), ("b","banana")] should return True.
-- check_dict_case [("a","apple"), ("A","banana"), ("B","banana")] should return False.
-- check_dict_case [("a","apple"), ("8","banana"), ("a","apple")] should return False.
-- check_dict_case [("Name","John"), ("Age","36"), ("City","Houston")] should return False.
-- check_dict_case [("STATE","NC"), ("ZIP","12345")] should return True.

check_dict_case :: [(String, String)] -> Bool
check_dict_case dict =
  length dict >  0
    && all (\x ->  all isUpper x) keys
    || all
      (\x ->  all isLower x)
      keys
  where
    keys = map  fst dict

-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (check_dict_case [("p","pineapple"),("b","banana")] == True)
    check (check_dict_case [("p","pineapple"),("A","banana"),("B","banana")] == False)
    check (check_dict_case [("p","pineapple"),("8","banana"),("a","apple")] == False)
    check (check_dict_case [("Name","John"),("Age","36"),("City","Houston")] == False)
    check (check_dict_case [("STATE","NC"),("ZIP","12345")] == True)
    check (check_dict_case [] == False)
