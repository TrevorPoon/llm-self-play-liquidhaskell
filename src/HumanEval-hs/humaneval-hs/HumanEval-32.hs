-- Task ID: HumanEval/32
-- Assigned To: Author A

-- Python Implementation:

-- import math
-- 
-- 
-- def poly(xs: list, x: float):
--     """
--     Evaluates polynomial with coefficients xs at point x.
--     return xs[0] + xs[1] * x + xs[1] * x^2 + .... xs[n] * x^n
--     """
--     return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])
-- 
-- 
-- def find_zero(xs: list):
--     """ xs are coefficients of a polynomial.
--     find_zero find x such that poly(x) = 0.
--     find_zero returns only only zero point, even if there are many.
--     Moreover, find_zero only takes list xs having even number of coefficients
--     and largest non zero coefficient as it guarantees
--     a solution.
--     >>> round(find_zero([1, 2]), 2) # f(x) = 1 + 2x
--     -0.5
--     >>> round(find_zero([-6, 11, -6, 1]), 2) # (x - 1) * (x - 2) * (x - 3) = -6 + 11x - 6x^2 + x^3
--     1.0
--     """
--     begin, end = -1., 1.
--     while poly(xs, begin) * poly(xs, end) > 0:
--         begin *= 2.0
--         end *= 2.0
--     while end - begin > 1e-10:
--         center = (begin + end) / 2.0
--         if poly(xs, center) * poly(xs, begin) > 0:
--             begin = center
--         else:
--             end = center
--     return begin
-- 


-- Haskell Implementation:
roundTo :: Double -> Int -> Double
roundTo x n =  (fromInteger $ round $  x * (10^n)) /  (10.0^^n)

-- Evaluates polynomial with coefficients xs at point x.
-- return xs[0] + xs[1] * x + xs[1] * x^2 + .... xs[n] * x^n
poly :: [Double] -> Double -> Double
poly coeffs x =  sum [coeff * x  ** fromIntegral i | (i, coeff) <-  zip [0..] coeffs]

-- xs are coefficients of a polynomial.
-- find_zero find x such that poly(x) = 0.
-- find_zero returns only only zero point, even if there are many.
-- Moreover, find_zero only takes list xs having even number of coefficients
-- and largest non zero coefficient as it guarantees
-- a solution.
-- >>> roundTo (find_zero [1,2]) 2 -- f(x) = 1 + 2x
-- -0.5
-- >>> roundTo (find_zero [-6,11,-6,1]) 2 -- (x - 1) * (x - 2) * (x - 3) = -6 + 11x - 6x^2 + x^3
-- 1.0
find_zero :: [Double] -> Double
find_zero coeffs = go (-1.0)  1.0
    where
        go :: Double -> Double -> Double
        go begin end
            | abs (end - begin) < 1e-10 =  begin
            | otherwise =
                let center =  (begin + end) / 2.0 in
                    if  poly coeffs center * poly coeffs begin  > 0
                        then  go center end
                        else  go begin center
