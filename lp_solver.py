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
                'success':      bool,   是否找到可行且最优的解
                'x':            ndarray, 最优解 (仅当 success=True 时有效)
                'fun':          float,   最优值 (原始目标函数值，sense 已处理)
                'status':       int,     scipy linprog 状态码
                'status_text':  str,     中文状态描述: 最优解/无可行解/无界/迭代次数超限/数值错误
                'message':      str,     求解器原始说明信息
                'nit':          int,     迭代次数
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

        return {
            "success": bool(result.success),
            "x": result.x.copy() if result.x is not None else None,
            "fun": fun_value,
            "status": status_code,
            "status_text": status_text,
            "message": result.message,
            "nit": int(result.nit) if getattr(result, "nit", None) is not None else 0,
        }


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
