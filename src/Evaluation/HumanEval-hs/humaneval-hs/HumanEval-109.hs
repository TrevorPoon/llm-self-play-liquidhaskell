-- Task ID: HumanEval/109
-- Assigned To: Author D

-- Python Implementation:

--
-- def move_one_ball(arr):
--     """We have an array 'arr' of N integers arr[1], arr[2], ..., arr[N].The
--     numbers in the array will be randomly ordered. Your task is to determine if
--     it is possible to get an array sorted in non-decreasing order by performing
--     the following operation on the given array:
--         You are allowed to perform right shift operation any number of times.
--
--     One right shift operation means shifting all elements of the array by one
--     position in the right direction. The last element of the array will be moved to
--     the starting position in the array i.e. 0th index.
--
--     If it is possible to obtain the sorted array by performing the above operation
--     then return True else return False.
--     If the given array is empty then return True.
--
--     Note: The given list is guaranteed to have unique elements.
--
--     For Example:
--
--     move_one_ball([3, 4, 5, 1, 2])==>True
--     Explanation: By performin 2 right shift operations, non-decreasing order can
--                  be achieved for the given array.
--     move_one_ball([3, 5, 4, 1, 2])==>False
--     Explanation:It is not possible to get non-decreasing order for the given
--                 array by performing any number of right shift operations.
--
--     """
--     if len(arr)==0:
--       return True
--     sorted_array=sorted(arr)
--     my_arr=[]
--
--     min_value=min(arr)
--     min_index=arr.index(min_value)
--     my_arr=arr[min_index:]+arr[0:min_index]
--     for i in range(len(arr)):
--       if my_arr[i]!=sorted_array[i]:
--         return False
--     return True
--

-- Haskell Implementation:

-- We have an array 'arr' of N integers arr[1], arr[2], ..., arr[N].The
-- numbers in the array will be randomly ordered. Your task is to determine if
-- it is possible to get an array sorted in non-decreasing order by performing
-- the following operation on the given array:
--     You are allowed to perform right shift operation any number of times.
--
-- One right shift operation means shifting all elements of the array by one
-- position in the right direction. The last element of the array will be moved to
-- the starting position in the array i.e. 0th index.
--
-- If it is possible to obtain the sorted array by performing the above operation
-- then return True else return False.
-- If the given array is empty then return True.
--
-- Note: The given list is guaranteed to have unique elements.
--
-- For Example:
--
-- move_one_ball [3, 4, 5, 1, 2] ==>True
-- Explanation: By performin 2 right shift operations, non-decreasing order can
--              be achieved for the given array.
-- move_one_ball [3, 5, 4, 1, 2] ==>False
-- Explanation:It is not possible to get non-decreasing order for the given
--             array by performing any number of right shift operations.
import Data.List (sort)

move_one_ball :: (Ord a) => [a] -> Bool
move_one_ball [] = True
move_one_ball arr = any (\i ->  rotate i arr ==  sort arr) [0 ..  length arr - 1]
  where
    rotate n xs =  drop n xs ++  take n xs
