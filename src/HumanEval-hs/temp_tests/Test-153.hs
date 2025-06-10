-- Task ID: HumanEval/153
-- Assigned To: Author B

-- Python Implementation:

--
-- def strongest_extension(class_name, extensions):
--     """You will be given the name of a class (a string) and a list of extensions.
--     The extensions are to be used to load additional classes to the class. The
--     strength of the extension is as follows: Let CAP be the number of the uppercase
--     letters in the extension's name, and let SM be the number of lowercase letters
--     in the extension's name, the strength is given by the fraction CAP - SM.
--     You should find the strongest extension and return a string in this
--     format: ClassName.StrongestExtensionName.
--     If there are two or more extensions with the same strength, you should
--     choose the one that comes first in the list.
--     For example, if you are given "Slices" as the class and a list of the
--     extensions: ['SErviNGSliCes', 'Cheese', 'StuFfed'] then you should
--     return 'Slices.SErviNGSliCes' since 'SErviNGSliCes' is the strongest extension
--     (its strength is -1).
--     Example:
--     for strongest_extension('my_class', ['AA', 'Be', 'CC']) == 'my_class.AA'
--     """
--     strong = extensions[0]
--     my_val = len([x for x in extensions[0] if x.isalpha() and x.isupper()]) - len([x for x in extensions[0] if x.isalpha() and x.islower()])
--     for s in extensions:
--         val = len([x for x in s if x.isalpha() and x.isupper()]) - len([x for x in s if x.isalpha() and x.islower()])
--         if val > my_val:
--             strong = s
--             my_val = val
--
--     ans = class_name + "." + strong
--     return ans
--
--

-- Haskell Implementation:

-- You will be given the name of a class (a string) and a list of extensions.
-- The extensions are to be used to load additional classes to the class. The
-- strength of the extension is as follows: Let CAP be the number of the uppercase
-- letters in the extension's name, and let SM be the number of lowercase letters
-- in the extension's name, the strength is given by the fraction CAP - SM.
-- You should find the strongest extension and return a string in this
-- format: ClassName.StrongestExtensionName.
-- If there are two or more extensions with the same strength, you should
-- choose the one that comes first in the list.
-- For example, if you are given "Slices" as the class and a list of the
-- extensions: ['SErviNGSliCes', 'Cheese', 'StuFfed'] then you should
-- return 'Slices.SErviNGSliCes' since 'SErviNGSliCes' is the strongest extension
-- (its strength is -1).
-- Example:
-- >>> strongest_extension "my_class" ["AA", "Be", "CC"]
-- "my_class.AA"

import Data.Char (isAlpha, isLower, isUpper)

strongest_extension :: String -> [String] -> String
strongest_extension class_name extensions =  class_name ++ "." ++ strongest
  where
    strongest :: String
    strongest =  head $ filter (\x ->  strength x == maximum (map strength extensions)) extensions
    strength x =  length [x | x <- x, isAlpha x, isUpper x] -  length [x | x <- x, isAlpha x, isLower x]



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (strongest_extension "Watashi" ["tEN","niNE","eIGHt8OKe"] == "Watashi.eIGHt8OKe")
    check (strongest_extension "Boku123" ["nani","NazeDa","YEs.WeCaNe","32145tggg"] == "Boku123.YEs.WeCaNe")
    check (strongest_extension "__YESIMHERE" ["t","eMptY","nothing","zeR00","NuLl__","123NoooneB321"] == "__YESIMHERE.NuLl__")
    check (strongest_extension "K" ["Ta","TAR","t234An","cosSo"] == "K.TAR")
    check (strongest_extension "__HAHA" ["Tab","123","781345","-_-"] == "__HAHA.123")
    check (strongest_extension "YameRore" ["HhAas","okIWILL123","WorkOut","Fails","-_-"] == "YameRore.okIWILL123")
    check (strongest_extension "finNNalLLly" ["Die","NowW","Wow","WoW"] == "finNNalLLly.WoW")
    check (strongest_extension "_" ["Bb","91245"] == "_.Bb")
    check (strongest_extension "Sp" ["671235","Bb"] == "Sp.671235")
