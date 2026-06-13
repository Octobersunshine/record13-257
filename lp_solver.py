import numpy as np
from scipy.optimize import linprog


_STATUS_TEXT_MAP = {
    0: "最优解",
    1: "迭代次数超限",
    2: "无可行解",
    3: "无界",
    4: "数值错误",
}


def _get_status_text(status_code, message):
    if status_code in _STATUS_TEXT_MAP:
        return _STATUS_TEXT_MAP[status_code]
    return f"未知状态 (status={status_code})"


def _safe_array(obj):
    if obj is None:
        return None
    arr = np.asarray(obj, dtype=np.float64)
    return arr.copy() if arr.size > 0 else None


def _shadow_price_sign_correction(marginals, sense):
    if marginals is None or len(marginals) == 0:
        return None
    if sense == "max":
        return -marginals.copy()
    return marginals.copy()


class LPSolver:
    """
    线性规划求解服务

    标准形式 (scipy linprog 形式):
        最小化: c @ x
        约束:
            A_ub @ x <= b_ub  (不等式约束)
            A_eq @ x == b_eq  (等式约束)
            lb <= x <= ub     (变量边界)
    """

    def __init__(self, method="highs"):
        valid_methods = ("highs", "highs-ds", "highs-ipm", "simplex", "interior-point")
        if method not in valid_methods:
            raise ValueError(f"method 必须是 {valid_methods} 之一，当前值: {method}")
        self.method = method

    @staticmethod
    def _to_ndarray(obj, name, expected_ndim=None):
        if obj is None:
            return None
        arr = np.asarray(obj, dtype=np.float64)
        if expected_ndim is not None and arr.ndim != expected_ndim:
            raise ValueError(f"{name} 必须是 {expected_ndim} 维数组，当前维度: {arr.ndim}")
        return arr

    @staticmethod
    def _validate_shapes(c, A_ub, b_ub, A_eq, b_eq, n_vars):
        if A_ub is not None:
            if A_ub.shape[1] != n_vars:
                raise ValueError(
                    f"A_ub 列数 ({A_ub.shape[1]}) 与变量数 ({n_vars}) 不匹配"
                )
            if b_ub is None or b_ub.shape[0] != A_ub.shape[0]:
                raise ValueError(
                    f"b_ub 长度 ({b_ub.shape[0] if b_ub is not None else 'None'}) "
                    f"与 A_ub 行数 ({A_ub.shape[0]}) 不匹配"
                )
        if A_eq is not None:
            if A_eq.shape[1] != n_vars:
                raise ValueError(
                    f"A_eq 列数 ({A_eq.shape[1]}) 与变量数 ({n_vars}) 不匹配"
                )
            if b_eq is None or b_eq.shape[0] != A_eq.shape[0]:
                raise ValueError(
                    f"b_eq 长度 ({b_eq.shape[0] if b_eq is not None else 'None'}) "
                    f"与 A_eq 行数 ({A_eq.shape[0]}) 不匹配"
                )

    @staticmethod
    def _normalize_bounds(bounds, n_vars):
        if bounds is None:
            return [(0, None)] * n_vars
        if len(bounds) != n_vars:
            raise ValueError(
                f"bounds 长度 ({len(bounds)}) 与变量数 ({n_vars}) 不匹配"
            )
        normalized = []
        for i, b in enumerate(bounds):
            if b is None:
                normalized.append((0, None))
                continue
            if not isinstance(b, (tuple, list)) or len(b) != 2:
                raise ValueError(f"bounds[{i}] 必须是 (lb, ub) 形式的二元组/列表")
            lb, ub = b
            if lb is not None and ub is not None and lb > ub:
                raise ValueError(f"bounds[{i}] 下界 ({lb}) 大于上界 ({ub})")
            normalized.append((lb, ub))
        return normalized

    def solve(
        self,
        c,
        A_ub=None,
        b_ub=None,
        A_eq=None,
        b_eq=None,
        bounds=None,
        sense="min",
    ):
        """
        求解线性规划问题

        参数:
            c:       目标函数系数 (1D array-like)，长度 = 变量数
            A_ub:    不等式约束系数矩阵 (2D array-like)，每行一个 <= 约束
            b_ub:    不等式约束右端 (1D array-like)，A_ub @ x <= b_ub
            A_eq:    等式约束系数矩阵 (2D array-like)，每行一个 == 约束
            b_eq:    等式约束右端 (1D array-like)，A_eq @ x == b_eq
            bounds:  变量边界列表，每个元素为 (lb, ub) 二元组；lb/ub 为 None 表示无界
                     默认每个变量 x_i >= 0
            sense:   优化方向: 'min' (默认) 或 'max'

        返回:
            dict: {
                'success':      bool,     是否找到可行且最优的解
                'x':            ndarray,  最优解 (仅当 success=True 时有效)
                'fun':          float,    最优值 (原始目标函数值，sense 已处理)
                'status':       int,      scipy linprog 状态码
                'status_text':  str,      中文状态描述: 最优解/无可行解/无界/迭代次数超限/数值错误
                'message':      str,      求解器原始说明信息
                'nit':          int,      迭代次数
                'sensitivity':  dict|None, 灵敏度分析信息 (仅当 success=True 且方法为 highs 系列时有效)
                    {
                        'shadow_prices': {
                            'ineq': ndarray|None,  不等式约束影子价格 (b_ub 每单位变化对最优值的影响)
                            'eq':   ndarray|None,  等式约束影子价格 (b_eq 每单位变化对最优值的影响)
                        },
                        'slack': {
                            'ineq': ndarray|None,  不等式约束松弛量
                            'eq':   ndarray|None,  等式约束残差
                        },
                        'reduced_costs': {
                            'lower': ndarray|None, 变量下界约简成本
                            'upper': ndarray|None, 变量上界约简成本
                        },
                    }
            }
        """
        if sense not in ("min", "max"):
            raise ValueError("sense 必须是 'min' 或 'max'")

        c_arr = self._to_ndarray(c, "c", expected_ndim=1)
        if c_arr.size == 0:
            raise ValueError("目标函数系数 c 不能为空")
        n_vars = c_arr.size

        A_ub_arr = self._to_ndarray(A_ub, "A_ub", expected_ndim=2) if A_ub is not None else None
        b_ub_arr = self._to_ndarray(b_ub, "b_ub", expected_ndim=1) if b_ub is not None else None
        A_eq_arr = self._to_ndarray(A_eq, "A_eq", expected_ndim=2) if A_eq is not None else None
        b_eq_arr = self._to_ndarray(b_eq, "b_eq", expected_ndim=1) if b_eq is not None else None

        self._validate_shapes(c_arr, A_ub_arr, b_ub_arr, A_eq_arr, b_eq_arr, n_vars)

        bounds_norm = self._normalize_bounds(bounds, n_vars)

        c_solver = c_arr if sense == "min" else -c_arr

        result = linprog(
            c_solver,
            A_ub=A_ub_arr,
            b_ub=b_ub_arr,
            A_eq=A_eq_arr,
            b_eq=b_eq_arr,
            bounds=bounds_norm,
            method=self.method,
        )

        status_code = int(result.status)
        status_text = _get_status_text(status_code, result.message)

        if result.fun is not None:
            fun_value = float(result.fun) if sense == "min" else float(-result.fun)
        else:
            fun_value = None

        sensitivity = None
        if result.success:
            sensitivity = self._extract_sensitivity(result, sense)

        return {
            "success": bool(result.success),
            "x": result.x.copy() if result.x is not None else None,
            "fun": fun_value,
            "status": status_code,
            "status_text": status_text,
            "message": result.message,
            "nit": int(result.nit) if getattr(result, "nit", None) is not None else 0,
            "sensitivity": sensitivity,
        }

    @staticmethod
    def _extract_sensitivity(result, sense):
        ineq_marginals = _safe_array(getattr(result.ineqlin, "marginals", None))
        eq_marginals = _safe_array(getattr(result.eqlin, "marginals", None))
        lower_marginals = _safe_array(getattr(result.lower, "marginals", None))
        upper_marginals = _safe_array(getattr(result.upper, "marginals", None))

        ineq_slack = _safe_array(getattr(result.ineqlin, "residual", None))
        eq_residual = _safe_array(getattr(result.eqlin, "residual", None))

        sp_ineq = _shadow_price_sign_correction(ineq_marginals, sense)
        sp_eq = _shadow_price_sign_correction(eq_marginals, sense)
        rc_lower = _shadow_price_sign_correction(lower_marginals, sense)
        rc_upper = _shadow_price_sign_correction(upper_marginals, sense)

        return {
            "shadow_prices": {
                "ineq": sp_ineq,
                "eq": sp_eq,
            },
            "slack": {
                "ineq": ineq_slack,
                "eq": eq_residual,
            },
            "reduced_costs": {
                "lower": rc_lower,
                "upper": rc_upper,
            },
        }

    def _solve_single(self, c_solver, A_ub_arr, b_ub_arr, A_eq_arr, b_eq_arr, bounds_norm):
        return linprog(
            c_solver,
            A_ub=A_ub_arr,
            b_ub=b_ub_arr,
            A_eq=A_eq_arr,
            b_eq=b_eq_arr,
            bounds=bounds_norm,
            method=self.method,
        )

    def sensitivity_analysis(
        self,
        c,
        A_ub=None,
        b_ub=None,
        A_eq=None,
        b_eq=None,
        bounds=None,
        sense="min",
        compute_ranges=False,
        range_step=None,
        range_max_iter=50,
    ):
        """
        完整的灵敏度分析

        参数:
            c, A_ub, b_ub, A_eq, b_eq, bounds, sense: 同 solve()
            compute_ranges: bool, 是否计算约束右端灵敏度范围 (默认 False，较耗时)
            range_step:   float, 范围搜索的步长 (默认自动选择 b 的 1%)
            range_max_iter: int, 范围搜索最大迭代次数

        返回:
            dict: {
                'success': bool,
                'x': ndarray,
                'fun': float,
                'status': int,
                'status_text': str,
                'message': str,
                'nit': int,
                'sensitivity': {
                    'shadow_prices': {'ineq': ..., 'eq': ...},
                    'slack':         {'ineq': ..., 'eq': ...},
                    'reduced_costs': {'lower': ..., 'upper': ...},
                    'ranges': {  # 仅当 compute_ranges=True 时存在
                        'ineq': [{'allowable_decrease': ..., 'allowable_increase': ...}, ...],
                        'eq':   [{'allowable_decrease': ..., 'allowable_increase': ...}, ...],
                    }
                }
            }
        """
        base_result = self.solve(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            sense=sense,
        )

        if not base_result["success"] or base_result["sensitivity"] is None:
            return base_result

        if not compute_ranges:
            return base_result

        c_arr = self._to_ndarray(c, "c", expected_ndim=1)
        n_vars = c_arr.size
        A_ub_arr = self._to_ndarray(A_ub, "A_ub", expected_ndim=2) if A_ub is not None else None
        b_ub_arr = self._to_ndarray(b_ub, "b_ub", expected_ndim=1) if b_ub is not None else None
        A_eq_arr = self._to_ndarray(A_eq, "A_eq", expected_ndim=2) if A_eq is not None else None
        b_eq_arr = self._to_ndarray(b_eq, "b_eq", expected_ndim=1) if b_eq is not None else None
        bounds_norm = self._normalize_bounds(bounds, n_vars)
        c_solver = c_arr if sense == "min" else -c_arr

        base_x = base_result["x"]
        base_fun = base_result["fun"]

        sp_ineq = base_result["sensitivity"]["shadow_prices"]["ineq"]
        sp_eq = base_result["sensitivity"]["shadow_prices"]["eq"]

        def _get_shadow_price(result, idx, is_ineq):
            if not result.success:
                return None
            sens = self._extract_sensitivity(result, sense)
            if is_ineq:
                if sens["shadow_prices"]["ineq"] is None or idx >= len(sens["shadow_prices"]["ineq"]):
                    return None
                return sens["shadow_prices"]["ineq"][idx]
            else:
                if sens["shadow_prices"]["eq"] is None or idx >= len(sens["shadow_prices"]["eq"]):
                    return None
                return sens["shadow_prices"]["eq"][idx]

        def _is_same_basis(result, idx, is_ineq, base_sp_val):
            sp_val = _get_shadow_price(result, idx, is_ineq)
            if sp_val is None:
                return False
            if base_sp_val == 0:
                return abs(sp_val) < 1e-6
            return np.isclose(sp_val, base_sp_val, rtol=1e-4, atol=1e-6)

        def _find_range(b_arr, idx, is_ineq=True):
            if b_arr is None:
                return None
            b_val = b_arr[idx]
            if is_ineq:
                base_sp = sp_ineq[idx] if sp_ineq is not None and idx < len(sp_ineq) else 0.0
            else:
                base_sp = sp_eq[idx] if sp_eq is not None and idx < len(sp_eq) else 0.0

            if range_step is None:
                step = max(abs(b_val) * 0.01, 1e-3)
            else:
                step = range_step

            allowable_increase = 0.0
            current_step = step
            b_test = b_val + current_step
            for _ in range(range_max_iter):
                b_new = b_arr.copy()
                b_new[idx] = b_test
                if is_ineq:
                    r = self._solve_single(c_solver, A_ub_arr, b_new, A_eq_arr, b_eq_arr, bounds_norm)
                else:
                    r = self._solve_single(c_solver, A_ub_arr, b_ub_arr, A_eq_arr, b_new, bounds_norm)
                if not _is_same_basis(r, idx, is_ineq, base_sp):
                    break
                allowable_increase = b_test - b_val
                current_step *= 1.5
                if current_step > 1e10:
                    allowable_increase = float("inf")
                    break
                b_test = b_val + current_step

            current_step = step
            allowable_decrease = 0.0
            b_test = b_val - current_step
            for _ in range(range_max_iter):
                b_new = b_arr.copy()
                b_new[idx] = b_test
                if is_ineq:
                    r = self._solve_single(c_solver, A_ub_arr, b_new, A_eq_arr, b_eq_arr, bounds_norm)
                else:
                    r = self._solve_single(c_solver, A_ub_arr, b_ub_arr, A_eq_arr, b_new, bounds_norm)
                if not _is_same_basis(r, idx, is_ineq, base_sp):
                    break
                allowable_decrease = b_val - b_test
                current_step *= 1.5
                if current_step > 1e10:
                    allowable_decrease = float("inf")
                    break
                b_test = b_val - current_step

            return {
                "allowable_decrease": allowable_decrease,
                "allowable_increase": allowable_increase,
            }

        ranges = {"ineq": [], "eq": []}

        if A_ub_arr is not None:
            for i in range(A_ub_arr.shape[0]):
                ranges["ineq"].append(_find_range(b_ub_arr, i, is_ineq=True))
        else:
            ranges["ineq"] = None

        if A_eq_arr is not None:
            for i in range(A_eq_arr.shape[0]):
                ranges["eq"].append(_find_range(b_eq_arr, i, is_ineq=False))
        else:
            ranges["eq"] = None

        base_result["sensitivity"]["ranges"] = ranges
        return base_result


def solve_lp(
    c,
    A_ub=None,
    b_ub=None,
    A_eq=None,
    b_eq=None,
    bounds=None,
    sense="min",
    method="highs",
):
    """
    便捷函数：直接求解线性规划问题 (无需实例化 LPSolver)

    参数与返回值同 LPSolver.solve
    """
    solver = LPSolver(method=method)
    return solver.solve(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        sense=sense,
    )


def sensitivity_analysis_lp(
    c,
    A_ub=None,
    b_ub=None,
    A_eq=None,
    b_eq=None,
    bounds=None,
    sense="min",
    method="highs",
    compute_ranges=False,
    range_step=None,
    range_max_iter=50,
):
    """
    便捷函数：直接进行灵敏度分析 (无需实例化 LPSolver)

    参数与返回值同 LPSolver.sensitivity_analysis
    """
    solver = LPSolver(method=method)
    return solver.sensitivity_analysis(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        sense=sense,
        compute_ranges=compute_ranges,
        range_step=range_step,
        range_max_iter=range_max_iter,
    )
