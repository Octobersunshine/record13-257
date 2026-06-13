import unittest
import numpy as np
from lp_solver import LPSolver, solve_lp


class TestLPSolver(unittest.TestCase):
    def setUp(self):
        self.solver = LPSolver()

    def test_basic_minimization(self):
        """
        最小化 z = -x1 - 4x2
        s.t.
             x1 + 2x2 <= 8
             x1 +  x2 <= 5
             x1, x2 >= 0
        顶点: (0,0)=0, (0,4)=-16, (2,3)=-14, (5,0)=-5
        最优解: x1=0, x2=4, 最优值 z=-16
        """
        c = [-1, -4]
        A_ub = [[1, 2], [1, 1]]
        b_ub = [8, 5]
        result = self.solver.solve(c=c, A_ub=A_ub, b_ub=b_ub)
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["x"][0], 0, places=5)
        self.assertAlmostEqual(result["x"][1], 4, places=5)
        self.assertAlmostEqual(result["fun"], -16, places=5)

    def test_maximization(self):
        """
        最大化 z = 3x1 + 2x2
        s.t.
             x1 + x2 <= 4
             2x1 + x2 <= 5
             x1, x2 >= 0
        最优解: x1=1, x2=3, 最优值 z=9
        """
        c = [3, 2]
        A_ub = [[1, 1], [2, 1]]
        b_ub = [4, 5]
        result = self.solver.solve(c=c, A_ub=A_ub, b_ub=b_ub, sense="max")
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["x"][0], 1, places=5)
        self.assertAlmostEqual(result["x"][1], 3, places=5)
        self.assertAlmostEqual(result["fun"], 9, places=5)

    def test_with_equality_constraint(self):
        """
        最小化 z = 3x1 + x2
        s.t.
             x1 + x2 = 6
             x1 - x2 >= 2  =>  -x1 + x2 <= -2
             x1, x2 >= 0
        解空间: x1 + x2 = 6 且 x1 - x2 >= 2, 即 x2 = 6 - x1, x1 >= 4
        z = 3x1 + (6-x1) = 2x1 + 6, 在 x1 最小时取极小
        最优解: x1=4, x2=2, 最优值 z=14
        """
        c = [3, 1]
        A_eq = [[1, 1]]
        b_eq = [6]
        A_ub = [[-1, 1]]
        b_ub = [-2]
        result = self.solver.solve(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["x"][0], 4, places=5)
        self.assertAlmostEqual(result["x"][1], 2, places=5)
        self.assertAlmostEqual(result["fun"], 14, places=5)

    def test_custom_bounds(self):
        """
        最小化 z = 2x1 + 3x2
        s.t.
             x1 + x2 >= 5
             x1, x2 无符号限制 (bounds=(None, None))
        转化: -x1 - x2 <= -5
        最优解: 让 x2 = 5 - x1, z = 2x1 + 3(5-x1) = 15 - x1  -> 最小需要 x1 尽可能大,
        但这里 x 无界，所以我们改成 x1 + x2 = 5 的等价问题验证自定义 bounds
        """
        c = [2, 3]
        A_eq = [[1, 1]]
        b_eq = [5]
        bounds = [(0, None), (0, 10)]
        result = self.solver.solve(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["x"][0], 5, places=5)
        self.assertAlmostEqual(result["x"][1], 0, places=5)
        self.assertAlmostEqual(result["fun"], 10, places=5)

    def test_negative_variable_bounds(self):
        """
        最小化 z = x1 - x2
        s.t.
             x1 + x2 <= 10
             x1 >= -5
             x2 >= -3
        最优解: x1=-5, x2=15? 但 x2 没有上限？加约束 x2 <= 20
        """
        c = [1, -1]
        A_ub = [[1, 1]]
        b_ub = [10]
        bounds = [(-5, None), (-3, 20)]
        result = self.solver.solve(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["x"][0], -5, places=5)
        self.assertAlmostEqual(result["x"][1], 15, places=5)
        self.assertAlmostEqual(result["fun"], -5 - 15, places=5)

    def test_infeasible_problem(self):
        """
        不可行问题:
        最小化 z = x1 + x2
        s.t.
             x1 + x2 <= 1
             x1 + x2 >= 3
             x1, x2 >= 0
        """
        c = [1, 1]
        A_ub = [[1, 1], [-1, -1]]
        b_ub = [1, -3]
        result = self.solver.solve(c=c, A_ub=A_ub, b_ub=b_ub)
        self.assertFalse(result["success"])

    def test_unbounded_problem(self):
        """
        无界问题:
        最大化 z = x1 + x2
        s.t.
             x1 - x2 >= 0  => -x1 + x2 <= 0
             x1, x2 >= 0
        无界: 让 x1=x2=t, z=2t, t->+inf
        """
        c = [1, 1]
        A_ub = [[-1, 1]]
        b_ub = [0]
        result = self.solver.solve(c=c, A_ub=A_ub, b_ub=b_ub, sense="max")
        self.assertFalse(result["success"])

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            LPSolver(method="invalid")

    def test_invalid_sense(self):
        with self.assertRaises(ValueError):
            self.solver.solve(c=[1, 2], sense="invalid")

    def test_empty_c(self):
        with self.assertRaises(ValueError):
            self.solver.solve(c=[])

    def test_dimension_mismatch_Aub(self):
        with self.assertRaises(ValueError):
            self.solver.solve(c=[1, 2], A_ub=[[1, 2, 3]], b_ub=[1])

    def test_dimension_mismatch_bub(self):
        with self.assertRaises(ValueError):
            self.solver.solve(c=[1, 2], A_ub=[[1, 2], [3, 4]], b_ub=[1])

    def test_bounds_length_mismatch(self):
        with self.assertRaises(ValueError):
            self.solver.solve(c=[1, 2], bounds=[(0, None)])

    def test_bounds_invalid_tuple(self):
        with self.assertRaises(ValueError):
            self.solver.solve(c=[1, 2], bounds=[(0, 1), (2,)])

    def test_bounds_lb_gt_ub(self):
        with self.assertRaises(ValueError):
            self.solver.solve(c=[1, 2], bounds=[(5, 1), (0, None)])

    def test_convenience_function_solve_lp(self):
        """测试便捷函数 solve_lp"""
        c = [-1, -4]
        A_ub = [[1, 2], [1, 1]]
        b_ub = [8, 5]
        result = solve_lp(c=c, A_ub=A_ub, b_ub=b_ub)
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["x"][0], 0, places=5)
        self.assertAlmostEqual(result["x"][1], 4, places=5)
        self.assertAlmostEqual(result["fun"], -16, places=5)

    def test_numpy_array_input(self):
        """测试 numpy 数组作为输入"""
        c = np.array([-1.0, -4.0])
        A_ub = np.array([[1.0, 2.0], [1.0, 1.0]])
        b_ub = np.array([8.0, 5.0])
        result = self.solver.solve(c=c, A_ub=A_ub, b_ub=b_ub)
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["x"][0], 0, places=5)
        self.assertAlmostEqual(result["x"][1], 4, places=5)

    def test_result_structure(self):
        """测试返回结果结构完整性"""
        c = [1, 2]
        A_eq = [[1, 1]]
        b_eq = [1]
        result = self.solver.solve(c=c, A_eq=A_eq, b_eq=b_eq)
        self.assertIn("success", result)
        self.assertIn("x", result)
        self.assertIn("fun", result)
        self.assertIn("message", result)
        self.assertIn("status", result)
        self.assertIn("nit", result)
        self.assertIsInstance(result["success"], bool)
        self.assertIsInstance(result["fun"], float)
        self.assertIsInstance(result["status"], int)
        self.assertIsInstance(result["nit"], int)


if __name__ == "__main__":
    unittest.main()
