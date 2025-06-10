-- Task ID: HumanEval/28
-- Assigned To: Author A

-- Python Implementation:

-- from typing import List
-- 
-- 
-- def concatenate(strings: List[str]) -> str:
--     """ Concatenate list of strings into a single string
--     >>> concatenate([])
--     ''
--     >>> concatenate(['a', 'b', 'c'])
--     'abc'
--     """
--     return ''.join(strings)
-- 


-- Haskell Implementation:

-- Concatenate list of strings into a single string
concatenate :: [String] -> String
concatenate strings =  concat strings



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (concatenate [] == "")
    check (concatenate ["x","y","z"] == "xyz")
    check (concatenate ["x","y","z","w","k"] == "xyzwk")
