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
        self.assertEqual(result["status"], 2)
        self.assertEqual(result["status_text"], "无可行解")
        self.assertIsNone(result["x"])
        self.assertIsNone(result["fun"])

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
        self.assertEqual(result["status"], 3)
        self.assertEqual(result["status_text"], "无界")
        self.assertIsNone(result["x"])
        self.assertIsNone(result["fun"])

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
        self.assertIn("status", result)
        self.assertIn("status_text", result)
        self.assertIn("message", result)
        self.assertIn("nit", result)
        self.assertIsInstance(result["success"], bool)
        self.assertIsInstance(result["fun"], float)
        self.assertIsInstance(result["status"], int)
        self.assertIsInstance(result["status_text"], str)
        self.assertIsInstance(result["message"], str)
        self.assertIsInstance(result["nit"], int)

    def test_optimal_status_text(self):
        """测试成功求解时返回 status_text 为'最优解'"""
        c = [3, 2]
        A_ub = [[1, 1], [2, 1]]
        b_ub = [4, 5]
        result = self.solver.solve(c=c, A_ub=A_ub, b_ub=b_ub, sense="max")
        self.assertTrue(result["success"])
        self.assertEqual(result["status"], 0)
        self.assertEqual(result["status_text"], "最优解")

    def test_sensitivity_shadow_prices(self):
        """测试影子价格：约束右端变化对目标函数的边际影响"""
        c = [3, 2]
        A_ub = [[1, 1], [2, 1]]
        b_ub = [4, 5]
        result = self.solver.solve(c=c, A_ub=A_ub, b_ub=b_ub, sense="max")
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["sensitivity"])
        self.assertIn("shadow_prices", result["sensitivity"])
        sp = result["sensitivity"]["shadow_prices"]
        self.assertIsNotNone(sp["ineq"])
        self.assertEqual(len(sp["ineq"]), 2)
        self.assertAlmostEqual(sp["ineq"][0], 1.0, places=5)
        self.assertAlmostEqual(sp["ineq"][1], 1.0, places=5)

    def test_sensitivity_shadow_price_verification(self):
        """通过实际扰动验证影子价格的正确性"""
        c = [3, 2]
        A_ub = [[1, 1], [2, 1]]
        b_ub = [4, 5]
        result = self.solver.solve(c=c, A_ub=A_ub, b_ub=b_ub, sense="max")
        base_fun = result["fun"]
        sp = result["sensitivity"]["shadow_prices"]["ineq"]

        b_ub2 = [5, 5]
        result2 = self.solver.solve(c=c, A_ub=A_ub, b_ub=b_ub2, sense="max")
        actual_change = result2["fun"] - base_fun
        predicted_change = sp[0] * 1.0
        self.assertAlmostEqual(actual_change, predicted_change, places=5)

    def test_sensitivity_slack(self):
        """测试约束松弛量"""
        c = [3, 2]
        A_ub = [[1, 1], [2, 1]]
        b_ub = [10, 10]
        result = self.solver.solve(c=c, A_ub=A_ub, b_ub=b_ub, sense="max")
        self.assertTrue(result["success"])
        slack = result["sensitivity"]["slack"]["ineq"]
        self.assertIsNotNone(slack)
        self.assertEqual(len(slack), 2)

    def test_sensitivity_reduced_costs(self):
        """测试变量约简成本"""
        c = [3, 2]
        A_ub = [[1, 1], [2, 1]]
        b_ub = [4, 5]
        result = self.solver.solve(c=c, A_ub=A_ub, b_ub=b_ub, sense="max")
        self.assertTrue(result["success"])
        rc = result["sensitivity"]["reduced_costs"]
        self.assertIn("lower", rc)
        self.assertIn("upper", rc)
        self.assertIsNotNone(rc["lower"])
        self.assertEqual(len(rc["lower"]), 2)

    def test_sensitivity_infeasible_problem(self):
        """无可行解问题应返回 sensitivity=None"""
        c = [1, 1]
        A_ub = [[1, 1], [-1, -1]]
        b_ub = [1, -3]
        result = self.solver.solve(c=c, A_ub=A_ub, b_ub=b_ub)
        self.assertFalse(result["success"])
        self.assertIsNone(result["sensitivity"])

    def test_sensitivity_with_equality(self):
        """测试等式约束的影子价格"""
        c = [3, 1]
        A_eq = [[1, 1]]
        b_eq = [6]
        A_ub = [[-1, 1]]
        b_ub = [-2]
        result = self.solver.solve(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        self.assertTrue(result["success"])
        sp = result["sensitivity"]["shadow_prices"]
        self.assertIsNotNone(sp["eq"])
        self.assertEqual(len(sp["eq"]), 1)

    def test_sensitivity_analysis_method(self):
        """测试 sensitivity_analysis 方法返回基础灵敏度信息"""
        c = [3, 2]
        A_ub = [[1, 1], [2, 1]]
        b_ub = [4, 5]
        result = self.solver.sensitivity_analysis(
            c=c, A_ub=A_ub, b_ub=b_ub, sense="max"
        )
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["sensitivity"])
        self.assertIn("shadow_prices", result["sensitivity"])
        self.assertIn("slack", result["sensitivity"])
        self.assertIn("reduced_costs", result["sensitivity"])

    def test_sensitivity_analysis_with_ranges(self):
        """测试 sensitivity_analysis 方法计算范围"""
        c = [3, 2]
        A_ub = [[1, 1], [2, 1]]
        b_ub = [4, 5]
        result = self.solver.sensitivity_analysis(
            c=c, A_ub=A_ub, b_ub=b_ub, sense="max",
            compute_ranges=True, range_step=0.1, range_max_iter=50
        )
        self.assertTrue(result["success"])
        self.assertIn("ranges", result["sensitivity"])
        ranges = result["sensitivity"]["ranges"]
        self.assertIn("ineq", ranges)
        self.assertIsNotNone(ranges["ineq"])
        self.assertEqual(len(ranges["ineq"]), 2)
        self.assertIn("allowable_decrease", ranges["ineq"][0])
        self.assertIn("allowable_increase", ranges["ineq"][0])
        self.assertGreater(ranges["ineq"][0]["allowable_decrease"], 0)
        self.assertGreater(ranges["ineq"][0]["allowable_increase"], 0)

    def test_sensitivity_analysis_minimization(self):
        """测试最小化问题的灵敏度分析"""
        c = [1, 4]
        A_ub = [[1, 2], [1, 1]]
        b_ub = [8, 5]
        result = self.solver.sensitivity_analysis(
            c=c, A_ub=A_ub, b_ub=b_ub, sense="min"
        )
        self.assertTrue(result["success"])
        sp = result["sensitivity"]["shadow_prices"]["ineq"]
        self.assertIsNotNone(sp)

    def test_convenience_sensitivity_analysis_lp(self):
        """测试便捷函数 sensitivity_analysis_lp"""
        from lp_solver import sensitivity_analysis_lp
        c = [3, 2]
        A_ub = [[1, 1], [2, 1]]
        b_ub = [4, 5]
        result = sensitivity_analysis_lp(
            c=c, A_ub=A_ub, b_ub=b_ub, sense="max"
        )
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["sensitivity"])
        self.assertAlmostEqual(result["fun"], 9.0, places=5)


if __name__ == "__main__":
    unittest.main()
