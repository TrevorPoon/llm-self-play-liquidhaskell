-- Task ID: HumanEval/71
-- Assigned To: Author E

-- Python Implementation:

-- 
-- def triangle_area(a, b, c):
--     '''
--     Given the lengths of the three sides of a triangle. Return the area of
--     the triangle rounded to 2 decimal points if the three sides form a valid triangle. 
--     Otherwise return -1
--     Three sides make a valid triangle when the sum of any two sides is greater 
--     than the third side.
--     Example:
--     triangle_area(3, 4, 5) == 6.00
--     triangle_area(1, 2, 10) == -1
--     '''
--     if a + b <= c or a + c <= b or b + c <= a:
--         return -1 
--     s = (a + b + c)/2    
--     area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
--     area = round(area, 2)
--     return area
-- 


-- Haskell Implementation:

-- Given the lengths of the three sides of a triangle. Return the area of
-- the triangle rounded to 2 decimal points if the three sides form a valid triangle.
-- Otherwise return -1
-- Three sides make a valid triangle when the sum of any two sides is greater
-- than the third side.
-- Example:
-- triangle_area 3 4 5 == 6.00
-- triangle_area 1 2 10 == -1

triangle_area :: Double -> Double -> Double -> Double
triangle_area a b c =  if a + b <= c || a + c <= b || b + c <= a then -1 else  round' (sqrt (s * (s - a) * (s - b) * (s - c))) 2
                      where 
                        s :: Double
                        s =  (a + b + c) / 2
                        round' x n =  (fromInteger $ round $ x * (10^n)) / (10.0^^n)


-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (triangle_area 3 4 5 == 6.0)
    check (triangle_area 1 2 10 == -1)
    check (triangle_area 4 8 5 == 8.18)
    check (triangle_area 2 2 2 == 1.73)
    check (triangle_area 1 2 3 == -1)
    check (triangle_area 10 5 7 == 16.25)
    check (triangle_area 2 6 3 == -1)
    check (triangle_area 1 1 1 == 0.43)
    check (triangle_area 2 2 10 == -1)
