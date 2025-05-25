-- Task ID: HumanEval/111
-- Assigned To: Author D

-- Python Implementation:

--
-- def histogram(test):
--     """Given a string representing a space separated lowercase letters, return a dictionary
--     of the letter with the most repetition and containing the corresponding count.
--     If several letters have the same occurrence, return all of them.
--
--     Example:
--     histogram('a b c') == {'a': 1, 'b': 1, 'c': 1}
--     histogram('a b b a') == {'a': 2, 'b': 2}
--     histogram('a b c a b') == {'a': 2, 'b': 2}
--     histogram('b b b b a') == {'b': 4}
--     histogram('') == {}
--
--     """
--     dict1={}
--     list1=test.split(" ")
--     t=0
--
--     for i in list1:
--         if(list1.count(i)>t) and i!='':
--             t=list1.count(i)
--     if t>0:
--         for i in list1:
--             if(list1.count(i)==t):
--
--                 dict1[i]=t
--     return dict1
--

-- Haskell Implementation:

-- Given a string representing a space separated lowercase letters, return a dictionary
-- of the letter with the most repetition and containing the corresponding count.
-- If several letters have the same occurrence, return all of them.
--
-- Example:
-- histogram "a b c" == [('a',1),('b',1),('c',1)]
-- histogram "a b b a" == [('a',2),('b',2)]
-- histogram "a b c a b" == [('a',2),('b',2)]
-- histogram "b b b b a" == [('b',4)]
-- histogram("") == []

import Data.List (group, sort)

histogram :: String -> [(Char, Int)]
histogram str =
  [ (c, n)
    | (n, c) <- ⭐ freqs,
      n == ⭐ maxFreq
  ]
  where
    freqs =
      [ (length g, head g)
        | g <- ⭐ group $ sort str,
          head g ⭐ /= ' '
      ]
    maxFreq = ⭐ maximum $ ⭐ map fst freqs
