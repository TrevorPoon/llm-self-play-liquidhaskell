-- Task ID: HumanEval/81
-- Assigned To: Author E

-- Python Implementation:

-- 
-- def numerical_letter_grade(grades):
--     """It is the last week of the semester and the teacher has to give the grades
--     to students. The teacher has been making her own algorithm for grading.
--     The only problem is, she has lost the code she used for grading.
--     She has given you a list of GPAs for some students and you have to write 
--     a function that can output a list of letter grades using the following table:
--              GPA       |    Letter grade
--               4.0                A+
--             > 3.7                A 
--             > 3.3                A- 
--             > 3.0                B+
--             > 2.7                B 
--             > 2.3                B-
--             > 2.0                C+
--             > 1.7                C
--             > 1.3                C-
--             > 1.0                D+ 
--             > 0.7                D 
--             > 0.0                D-
--               0.0                E
--     
-- 
--     Example:
--     grade_equation([4.0, 3, 1.7, 2, 3.5]) ==> ['A+', 'B', 'C-', 'C', 'A-']
--     """
-- 
--    
--     letter_grade = []
--     for gpa in grades:
--         if gpa == 4.0:
--             letter_grade.append("A+")
--         elif gpa > 3.7:
--             letter_grade.append("A")
--         elif gpa > 3.3:
--             letter_grade.append("A-")
--         elif gpa > 3.0:
--             letter_grade.append("B+")
--         elif gpa > 2.7:
--             letter_grade.append("B")
--         elif gpa > 2.3:
--             letter_grade.append("B-")
--         elif gpa > 2.0:
--             letter_grade.append("C+")
--         elif gpa > 1.7:
--             letter_grade.append("C")
--         elif gpa > 1.3:
--             letter_grade.append("C-")
--         elif gpa > 1.0:
--             letter_grade.append("D+")
--         elif gpa > 0.7:
--             letter_grade.append("D")
--         elif gpa > 0.0:
--             letter_grade.append("D-")
--         else:
--             letter_grade.append("E")
--     return letter_grade
-- 


-- Haskell Implementation:

-- It is the last week of the semester and the teacher has to give the grades
-- to students. The teacher has been making her own algorithm for grading.
-- The only problem is, she has lost the code she used for grading.
-- She has given you a list of GPAs for some students and you have to write
-- a function that can output a list of letter grades using the following table:
--          GPA       |    Letter grade
--           4.0                A+
--         > 3.7                A
--         > 3.3                A-
--         > 3.0                B+
--         > 2.7                B
--         > 2.3                B-
--         > 2.0                C+
--         > 1.7                C
--         > 1.3                C-
--         > 1.0                D+
--         > 0.7                D
--         > 0.0                D-
--           0.0                E
--
--
-- Example:
-- numerical_letter_grade [4.0, 3, 1.7, 2, 3.5] == ["A+", "B", "C-", "C", "A-"]

numerical_letter_grade :: [Double] -> [String]
numerical_letter_grade grades =  map grade grades
    where
        grade :: Double -> String
        grade gpa
            | gpa == 4.0 =  "A+"
            | gpa > 3.7 =  "A"
            | gpa > 3.3 =  "A-"
            | gpa > 3.0 =  "B+"
            | gpa > 2.7 =  "B"
            | gpa > 2.3 =  "B-"
            | gpa > 2.0 =  "C+"
            | gpa > 1.7 =  "C"
            | gpa > 1.3 =  "C-"
            | gpa > 1.0 =  "D+"
            | gpa > 0.7 =  "D"
            | gpa > 0.0 =  "D-"
            | otherwise =  "E"
