-- Task ID: HumanEval/82
-- Assigned To: Author E

-- Python Implementation:

-- 
-- def prime_length(string):
--     """Write a function that takes a string and returns True if the string
--     length is a prime number or False otherwise
--     Examples
--     prime_length('Hello') == True
--     prime_length('abcdcba') == True
--     prime_length('kittens') == True
--     prime_length('orange') == False
--     """
--     l = len(string)
--     if l == 0 or l == 1:
--         return False
--     for i in range(2, l):
--         if l % i == 0:
--             return False
--     return True
-- 


-- Haskell Implementation:

-- Write a function that takes a string and returns True if the string
-- length is a prime number or False otherwise
-- Examples
-- prime_length "Hello" == True
-- prime_length "abcdcba" == True
-- prime_length "kittens" == True
-- prime_length "orange" == False

prime_length :: String -> Bool
prime_length string = let l =  length string
                      in l /= 0 && l /= 1 && null [i | i <-  [2..(l-1)], l `mod` i == 0]



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (prime_length "Hello"    == True)
    check (prime_length "abcdcba"  == True)
    check (prime_length "kittens"  == True)
    check (prime_length "orange"   == False)
    check (prime_length "wow"      == True)
    check (prime_length "world"    == True)
    check (prime_length "MadaM"    == True)
    check (prime_length "Wow"      == True)
    check (prime_length ""         == False)
    check (prime_length "HI"       == True)
    check (prime_length "go"       == True)
    check (prime_length "gogo"     == False)
    check (prime_length "aaaaaaaaaaaaaaa" == False)
    check (prime_length "Madam"    == True)
    check (prime_length "M"        == False)
    check (prime_length "0"        == False)
