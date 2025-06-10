-- Task ID: HumanEval/48
-- Assigned To: Author C

-- Python Implementation:

-- 
-- 
-- def is_palindrome(text: str):
--     """
--     Checks if given string is a palindrome
--     >>> is_palindrome('')
--     True
--     >>> is_palindrome('aba')
--     True
--     >>> is_palindrome('aaaaa')
--     True
--     >>> is_palindrome('zbcd')
--     False
--     """
--     for i in range(len(text)):
--         if text[i] != text[len(text) - 1 - i]:
--             return False
--     return True
-- 


-- Haskell Implementation:

-- Checks if given string is a palindrome
-- >>> is_palindrome ""
-- True
-- >>> is_palindrome "aba"
-- True
-- >>> is_palindrome "aaaaa"
-- True
-- >>> is_palindrome "zbcd"
-- False
is_palindrome :: String -> Bool
is_palindrome =  (==) <*> reverse



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (is_palindrome "" == True)
    check (is_palindrome "aba" == True)
    check (is_palindrome "aaaaa" == True)
    check (is_palindrome "zbcd" == False)
    check (is_palindrome "xywyx" == True)
    check (is_palindrome "xywyz" == False)
    check (is_palindrome "xywzx" == False)