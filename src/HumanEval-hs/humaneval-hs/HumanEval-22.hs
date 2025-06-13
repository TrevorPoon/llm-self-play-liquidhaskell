-- Task ID: HumanEval/22


-- Haskell Implementation:

import Text.Read  (readMaybe)
import Data.Maybe (mapMaybe)

-- Filter given list of strings only for integers.
-- >>> filter_integers(["a", "3.14", "5"])
-- [5]
-- >>> filter_integers(["1", "2", "3", "abc", "{}", "[]"])
-- [1, 2, 3]
filter_integers :: [String] -> [Int]
filter_integers = mapMaybe readMaybe