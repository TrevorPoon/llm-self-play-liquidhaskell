-- Task ID: HumanEval/136
-- Assigned To: Author B

-- Python Implementation:

--
-- def largest_smallest_integers(lst):
--     '''
--     Create a function that returns a tuple (a, b), where 'a' is
--     the largest of negative integers, and 'b' is the smallest
--     of positive integers in a list.
--     If there is no negative or positive integers, return them as None.
--
--     Examples:
--     largest_smallest_integers([2, 4, 1, 3, 5, 7]) == (None, 1)
--     largest_smallest_integers([]) == (None, None)
--     largest_smallest_integers([0]) == (None, None)
--     '''
--     smallest = list(filter(lambda x: x < 0, lst))
--     largest = list(filter(lambda x: x > 0, lst))
--     return (max(smallest) if smallest else None, min(largest) if largest else None)
--

-- Haskell Implementation:

-- Create a function that returns a tuple (a, b), where 'a' is
-- the largest of negative integers, and 'b' is the smallest
-- of positive integers in a list.
-- If there is no negative or positive integers, return them as None.
--
-- Examples:
-- >>> largest_smallest_integers [2, 4, 1, 3, 5, 7]
-- (Nothing,Just 1)
-- >>> largest_smallest_integers []
-- (Nothing,Nothing)
-- >>> largest_smallest_integers [0]
-- (Nothing,Nothing)

import Data.Maybe

largest_smallest_integers :: [Int] -> (Maybe Int, Maybe Int)
largest_smallest_integers [] =  (Nothing, Nothing)
largest_smallest_integers arr =  largest_smallest_integers' arr Nothing Nothing
  where
    largest_smallest_integers' :: [Int] -> Maybe Int -> Maybe Int -> (Maybe Int, Maybe Int)
    largest_smallest_integers' [] largest smallest = (largest, smallest)
    largest_smallest_integers' (x : xs) largest smallest
      | x < 0 && isNothing largest =  largest_smallest_integers' xs (Just x) smallest
      | x < 0 =  largest_smallest_integers' xs (Just  (max (fromJust largest) x)) smallest
      | x > 0 && isNothing smallest =  largest_smallest_integers' xs largest (Just x)
      | x > 0 =  largest_smallest_integers' xs largest (Just  (min (fromJust smallest) x))
      | otherwise =  largest_smallest_integers' xs largest smallest

-- Alternative using fold:
largest_smallest_integers_fold :: [Int] -> (Maybe Int, Maybe Int)
largest_smallest_integers_fold [] =  (Nothing, Nothing)
largest_smallest_integers_fold arr =  foldl largest_smallest_integers_fold' (Nothing, Nothing) arr
  where
    largest_smallest_integers_fold' :: (Maybe Int, Maybe Int) -> Int -> (Maybe Int, Maybe Int)
    largest_smallest_integers_fold' (largest, smallest) x
      | x < 0 && isNothing largest =  (Just x, smallest)
      | x < 0 =  (Just (max (fromJust largest) x), smallest)
      | x > 0 && isNothing smallest =  (largest, Just x)
      | x > 0 =  (largest, Just (min (fromJust smallest) x))
      | otherwise =  (largest, smallest)



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (largest_smallest_integers [2,4,1,3,5,7]   == (Nothing,Just 1))
    check (largest_smallest_integers [2,4,1,3,5,7,0] == (Nothing,Just 1))
    check (largest_smallest_integers [1,3,2,4,5,6,-2] == (Just (-2),Just 1))
    check (largest_smallest_integers [4,5,3,6,2,7,-7] == (Just (-7),Just 2))
    check (largest_smallest_integers [7,3,8,4,9,2,5,-9] == (Just (-9),Just 2))
    check (largest_smallest_integers []               == (Nothing,Nothing))
    check (largest_smallest_integers [0]              == (Nothing,Nothing))
    check (largest_smallest_integers [-1,-3,-5,-6]    == (Just (-1),Nothing))
    check (largest_smallest_integers [-1,-3,-5,-6,0]  == (Just (-1),Nothing))
    check (largest_smallest_integers [-6,-4,-4,-3,1]  == (Just (-3),Just 1))
    check (largest_smallest_integers [-6,-4,-4,-3,-100,1] == (Just (-3),Just 1))
