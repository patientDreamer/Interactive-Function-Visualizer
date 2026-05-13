import re
import math

import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import plotly.graph_objects as go

from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)


# ============================================================
# Page Setup
# ============================================================

st.set_page_config(
    page_title="Interactive Function Illustrator",
    layout="wide"
)

st.title("Interactive Function Illustrator")
st.caption(
    "A mini Desmos-like visualizer: functions, piecewise functions, restrictions, "
    "inequalities, tables, regressions, calculus features, and complex-valued functions."
)


# ============================================================
# Parser Setup
# ============================================================

TRANSFORMATIONS = (
    standard_transformations
    + (implicit_multiplication_application, convert_xor)
)


CUSTOM_COLORSCALES = {
    "Pink Purple Blue": [
        [0.0, "#ff5fa2"],
        [0.25, "#c77dff"],
        [0.5, "#7b2cbf"],
        [0.75, "#4361ee"],
        [1.0, "#4cc9f0"],
    ],
    "Cyber Neon": [
        [0.0, "#ff006e"],
        [0.3, "#8338ec"],
        [0.6, "#3a86ff"],
        [1.0, "#00f5d4"],
    ],
    "Ocean": [
        [0.0, "#03045e"],
        [0.3, "#0077b6"],
        [0.6, "#00b4d8"],
        [1.0, "#90e0ef"],
    ],
    "Sunset Glow": [
        [0.0, "#ff9a9e"],
        [0.25, "#fecfef"],
        [0.5, "#fcb69f"],
        [0.75, "#ff8177"],
        [1.0, "#b12a5b"],
    ],
    "Aurora": [
        [0.0, "#001219"],
        [0.2, "#005f73"],
        [0.45, "#0a9396"],
        [0.7, "#94d2bd"],
        [1.0, "#ee9b00"],
    ],
    "Galaxy": [
        [0.0, "#0b1026"],
        [0.2, "#3a0ca3"],
        [0.45, "#7209b7"],
        [0.7, "#f72585"],
        [1.0, "#4cc9f0"],
    ],
    "Cotton Candy": [
        [0.0, "#ffafcc"],
        [0.25, "#ffc8dd"],
        [0.5, "#cdb4db"],
        [0.75, "#bde0fe"],
        [1.0, "#a2d2ff"],
    ],
    "Emerald Night": [
        [0.0, "#081c15"],
        [0.25, "#1b4332"],
        [0.5, "#2d6a4f"],
        [0.75, "#52b788"],
        [1.0, "#b7e4c7"],
    ],
    "Fire Ice": [
        [0.0, "#03045e"],
        [0.25, "#0077b6"],
        [0.5, "#caf0f8"],
        [0.75, "#ffb703"],
        [1.0, "#d00000"],
    ],
    "Lavender Dream": [
        [0.0, "#240046"],
        [0.25, "#5a189a"],
        [0.5, "#9d4edd"],
        [0.75, "#c77dff"],
        [1.0, "#e0aaff"],
    ],
}


ALLOWED_SYMBOL_NAMES = [
    "x", "y", "z",
    "t", "u", "v",
    "a", "b", "c",
    "m", "n", "r",
    "theta"
]

ALLOWED_SYMBOLS = {
    name: sp.Symbol(name)
    for name in ALLOWED_SYMBOL_NAMES
}


ALLOWED_FUNCTIONS = {
    # Basic trig
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "sec": lambda x: 1 / sp.cos(x),
    "csc": lambda x: 1 / sp.sin(x),
    "cot": lambda x: 1 / sp.tan(x),

    # Inverse trig
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "arctan": sp.atan,

    # Hyperbolic trig
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,

    # Inverse hyperbolic trig
    "asinh": sp.asinh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,
    "arcsinh": sp.asinh,
    "arccosh": sp.acosh,
    "arctanh": sp.atanh,

    # Logs and exponentials
    "exp": sp.exp,
    "ln": sp.log,
    "log": sp.log,
    "log10": lambda x: sp.log(x, 10),

    # Other common functions
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "floor": sp.floor,
    "ceil": sp.ceiling,
    "ceiling": sp.ceiling,
    "gamma": sp.gamma,

    # Piecewise
    "Piecewise": sp.Piecewise,

    # Complex
    "I": sp.I,
    "i": sp.I,
    "re": sp.re,
    "im": sp.im,
    "arg": sp.arg,

    # Constants
    "pi": sp.pi,
    "e": sp.E,
    "E": sp.E,
    "True": True,
    "False": False,
}


FUNCTION_NAMES = [
    "arcsinh", "arccosh", "arctanh",
    "arcsin", "arccos", "arctan",
    "asinh", "acosh", "atanh",
    "sinh", "cosh", "tanh",
    "asin", "acos", "atan",
    "sin", "cos", "tan",
    "sec", "csc", "cot",
    "sqrt", "log10", "log", "ln", "exp",
    "abs", "floor", "ceil", "ceiling", "gamma"
]

FUNCTION_NAMES = sorted(FUNCTION_NAMES, key=len, reverse=True)


# ============================================================
# Safe Numeric Functions
# ============================================================

def safe_gamma_scalar(value):
    try:
        result = math.gamma(float(value))
        if math.isfinite(result):
            return result
        return np.nan
    except Exception:
        return np.nan


safe_gamma = np.vectorize(safe_gamma_scalar)


NUMERIC_MODULES = [
    {
        "gamma": safe_gamma,
        "floor": np.floor,
        "ceiling": np.ceil,
        "ceil": np.ceil,
        "Abs": np.abs,
        "abs": np.abs,
        "sec": lambda x: 1 / np.cos(x),
        "csc": lambda x: 1 / np.sin(x),
        "cot": lambda x: 1 / np.tan(x),
        "log": np.log,
        "ln": np.log,
        "log10": np.log10,
        "I": 1j,
        "i": 1j,
    },
    "numpy"
]


# ============================================================
# Text Normalization
# ============================================================

def split_top_level_commas(text: str):
    parts = []
    current = []
    depth_parentheses = 0
    depth_braces = 0

    for char in text:
        if char == "(":
            depth_parentheses += 1
        elif char == ")":
            depth_parentheses -= 1
        elif char == "{":
            depth_braces += 1
        elif char == "}":
            depth_braces -= 1

        if char == "," and depth_parentheses == 0 and depth_braces == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)

    if current:
        parts.append("".join(current))

    return parts


def split_function_input(user_input: str):
    """
    Supports:
        f(x) = sin(x)
        y = sin(x)
        y < x^2
        y <= x^2 {1<x<5}
        sin(x)
    """

    text = user_input.strip().replace(" ", "")

    declared_variables = None
    inequality_info = None

    inequality_match = re.match(r"^y(<=|>=|<|>)(.+)$", text)

    if inequality_match:
        operator = inequality_match.group(1)
        rhs = inequality_match.group(2)
        inequality_info = operator
        return declared_variables, rhs, inequality_info

    function_match = re.match(r"^[a-zA-Z]\((.*?)\)=(.+)$", text)

    if function_match:
        declared_variables = [
            item.strip()
            for item in function_match.group(1).split(",")
            if item.strip()
        ]
        rhs = function_match.group(2)
        return declared_variables, rhs, inequality_info

    y_match = re.match(r"^y=(.+)$", text)

    if y_match:
        rhs = y_match.group(1)
        return declared_variables, rhs, inequality_info

    return declared_variables, text, inequality_info


def extract_restriction(expression_text: str):
    """
    Extracts restrictions like:
        x^2{1<x<5}
        sin(x){x>0,y<1}

    Does not treat a leading { ... } piecewise function as a restriction.
    """

    text = expression_text.strip()

    if text.startswith("{"):
        return text, None

    match = re.match(r"^(.*?)\{(.+)\}$", text)

    if match:
        expression_part = match.group(1)
        restriction_part = match.group(2)
        return expression_part, restriction_part

    return text, None


def replace_desmos_piecewise(expression_text: str):
    """
    Converts:
        {x<0:-x,otherwise:x}

    into:
        Piecewise((-x,x<0),(x,True))
    """

    text = expression_text.strip()

    if not (text.startswith("{") and text.endswith("}")):
        return expression_text

    inner = text[1:-1]
    raw_pieces = split_top_level_commas(inner)

    pieces = []

    for piece in raw_pieces:
        if ":" not in piece:
            continue

        condition, expression = piece.split(":", 1)
        condition = condition.strip()
        expression = expression.strip()

        if condition.lower() in ["otherwise", "else", "true"]:
            condition = "True"

        pieces.append(f"({expression},{condition})")

    if not pieces:
        return expression_text

    return "Piecewise(" + ",".join(pieces) + ")"


def replace_abs_bars(expression_text: str):
    pattern = r"\|([^|]+)\|"

    while re.search(pattern, expression_text):
        expression_text = re.sub(pattern, r"abs(\1)", expression_text)

    return expression_text


def replace_function_powers(expression_text: str):
    trig_like = [
        "sin", "cos", "tan",
        "sec", "csc", "cot",
        "sinh", "cosh", "tanh",
        "asin", "acos", "atan",
        "asinh", "acosh", "atanh"
    ]

    trig_like = sorted(trig_like, key=len, reverse=True)

    for func in trig_like:
        expression_text = re.sub(
            rf"\b{func}\^(\d+)\(([^()]+)\)",
            rf"({func}(\2))^\1",
            expression_text
        )

        expression_text = re.sub(
            rf"\b{func}\^(\d+)([a-zA-Z])\b",
            rf"({func}(\2))^\1",
            expression_text
        )

    return expression_text


def replace_factorials(expression_text: str):
    pattern_func = r"([a-zA-Z]+\(.*?\))!"

    while re.search(pattern_func, expression_text):
        expression_text = re.sub(
            pattern_func,
            r"gamma((\1)+1)",
            expression_text
        )

    pattern_parentheses = r"\(([^()]+)\)!"

    while re.search(pattern_parentheses, expression_text):
        expression_text = re.sub(
            pattern_parentheses,
            r"gamma((\1)+1)",
            expression_text
        )

    pattern_simple = r"([a-zA-Z]|\d+\.?\d*)!"

    expression_text = re.sub(
        pattern_simple,
        r"gamma(\1+1)",
        expression_text
    )

    return expression_text


def replace_casual_function_inputs(expression_text: str):
    for func in FUNCTION_NAMES:
        expression_text = re.sub(
            rf"\b{func}([a-zA-Z])\b",
            rf"{func}(\1)",
            expression_text
        )

    return expression_text


def normalize_function_text(expression_text: str):
    expression_text = expression_text.strip().replace(" ", "")

    expression_text = replace_desmos_piecewise(expression_text)

    expression_text = expression_text.replace("arcsinh", "asinh")
    expression_text = expression_text.replace("arccosh", "acosh")
    expression_text = expression_text.replace("arctanh", "atanh")
    expression_text = expression_text.replace("arcsin", "asin")
    expression_text = expression_text.replace("arccos", "acos")
    expression_text = expression_text.replace("arctan", "atan")
    expression_text = expression_text.replace("ceiling", "ceil")

    expression_text = replace_abs_bars(expression_text)
    expression_text = replace_function_powers(expression_text)
    expression_text = replace_factorials(expression_text)
    expression_text = replace_casual_function_inputs(expression_text)

    return expression_text


def parse_function(user_input: str):
    declared_variables, expression_text, inequality_info = split_function_input(user_input)
    expression_text, restriction_text = extract_restriction(expression_text)
    normalized_text = normalize_function_text(expression_text)

    local_dict = {**ALLOWED_SYMBOLS, **ALLOWED_FUNCTIONS}

    expr = parse_expr(
        normalized_text,
        local_dict=local_dict,
        transformations=TRANSFORMATIONS,
        evaluate=True,
    )

    if declared_variables:
        variables = [
            ALLOWED_SYMBOLS[v]
            for v in declared_variables
            if v in ALLOWED_SYMBOLS
        ]
    else:
        variables = sorted(expr.free_symbols, key=lambda s: s.name)

    return expr, variables, normalized_text, restriction_text, inequality_info


# ============================================================
# Numeric Evaluation
# ============================================================

def evaluate_raw_1d(expr, var, x_vals):
    f = sp.lambdify(var, expr, modules=NUMERIC_MODULES)

    with np.errstate(all="ignore"):
        y_vals = f(x_vals)

    if np.isscalar(y_vals):
        y_vals = np.full_like(x_vals, y_vals, dtype=complex)

    return np.array(y_vals, dtype=np.complex128)


def clean_real_array(values):
    values = np.array(values, dtype=np.complex128)

    real_values = np.real(values)
    imaginary_values = np.imag(values)

    valid_mask = (
        np.isfinite(real_values)
        & np.isfinite(imaginary_values)
        & (np.abs(imaginary_values) < 1e-8)
    )

    cleaned = np.full(real_values.shape, np.nan)
    cleaned[valid_mask] = real_values[valid_mask]

    return cleaned


def complex_to_view(values, view_mode):
    values = np.array(values, dtype=np.complex128)

    if view_mode == "Real part":
        return np.real(values)

    if view_mode == "Imaginary part":
        return np.imag(values)

    if view_mode == "Magnitude":
        return np.abs(values)

    if view_mode == "Phase / Argument":
        return np.angle(values)

    return np.real(values)


def evaluate_expression_scalar(expr, var, x0):
    try:
        raw = evaluate_raw_1d(expr, var, np.array([float(x0)]))
        cleaned = clean_real_array(raw)[0]

        if np.isfinite(cleaned):
            return float(cleaned)

        return np.nan
    except Exception:
        return np.nan


def evaluate_expression_2d(expr, x_var, y_var, X, Y):
    f = sp.lambdify((x_var, y_var), expr, modules=NUMERIC_MODULES)

    with np.errstate(all="ignore"):
        Z = f(X, Y)

    if np.isscalar(Z):
        Z = np.full_like(X, Z, dtype=complex)

    return clean_real_array(Z)


# ============================================================
# Restrictions
# ============================================================

def apply_restrictions(x_vals, y_vals, restriction_text):
    """
    Supports common restrictions:
        1<x<5
        x>0
        x<=3
        y>0
        -2<y<2
        x>0,y<5
    """

    if not restriction_text:
        return y_vals

    mask = np.ones_like(x_vals, dtype=bool)

    restrictions = re.split(r",|&", restriction_text)

    for condition in restrictions:
        condition = condition.strip().replace(" ", "")

        if not condition:
            continue

        chain_match = re.match(
            r"^(-?\d+\.?\d*)(<=|<)(x|y)(<=|<)(-?\d+\.?\d*)$",
            condition
        )

        if chain_match:
            left_num = float(chain_match.group(1))
            var_name = chain_match.group(3)
            right_num = float(chain_match.group(5))

            target = x_vals if var_name == "x" else y_vals
            mask &= target > left_num
            mask &= target < right_num
            continue

        simple_match = re.match(
            r"^(x|y)(<=|>=|<|>)(-?\d+\.?\d*)$",
            condition
        )

        if simple_match:
            var_name = simple_match.group(1)
            operator = simple_match.group(2)
            number = float(simple_match.group(3))

            target = x_vals if var_name == "x" else y_vals

            if operator == "<":
                mask &= target < number
            elif operator == "<=":
                mask &= target <= number
            elif operator == ">":
                mask &= target > number
            elif operator == ">=":
                mask &= target >= number

    restricted_y = np.array(y_vals, dtype=float)
    restricted_y[~mask] = np.nan

    return restricted_y


# ============================================================
# Calculus Helpers
# ============================================================

def filter_close_points(points, tolerance=0.1):
    if not points:
        return []

    filtered = [points[0]]

    for point in points[1:]:
        if abs(point[0] - filtered[-1][0]) > tolerance:
            filtered.append(point)

    return filtered


def find_approx_local_extrema(x_vals, y_vals):
    valid = np.isfinite(y_vals)

    x = x_vals[valid]
    y = y_vals[valid]

    if len(x) < 5:
        return [], []

    dy = np.gradient(y, x)

    maxima = []
    minima = []

    for i in range(1, len(dy) - 1):
        left = dy[i - 1]
        right = dy[i + 1]

        if not np.isfinite(left) or not np.isfinite(right):
            continue

        if left > 0 and right < 0:
            maxima.append((x[i], y[i]))

        if left < 0 and right > 0:
            minima.append((x[i], y[i]))

    return filter_close_points(maxima), filter_close_points(minima)


def approximate_derivative_at(expr, var, x0):
    derivative_expr = sp.diff(expr, var)

    try:
        value = evaluate_expression_scalar(derivative_expr, var, x0)
        if np.isfinite(value):
            return value, derivative_expr
    except Exception:
        pass

    h = 1e-5
    y_plus = evaluate_expression_scalar(expr, var, x0 + h)
    y_minus = evaluate_expression_scalar(expr, var, x0 - h)

    if np.isfinite(y_plus) and np.isfinite(y_minus):
        return (y_plus - y_minus) / (2 * h), derivative_expr

    return np.nan, derivative_expr


def tangent_line_values(expr, var, x0, x_vals):
    y0 = evaluate_expression_scalar(expr, var, x0)
    slope, derivative_expr = approximate_derivative_at(expr, var, x0)

    if not np.isfinite(y0) or not np.isfinite(slope):
        return None, y0, slope, derivative_expr

    tangent_y = y0 + slope * (x_vals - x0)

    return tangent_y, y0, slope, derivative_expr


def analyze_continuity_and_differentiability(expr, var, x0):
    h = 1e-5

    y0 = evaluate_expression_scalar(expr, var, x0)
    y_left = evaluate_expression_scalar(expr, var, x0 - h)
    y_right = evaluate_expression_scalar(expr, var, x0 + h)

    if np.isfinite(y0) and np.isfinite(y_left) and np.isfinite(y_right):
        if abs(y_left - y0) < 1e-3 and abs(y_right - y0) < 1e-3:
            continuity = "Likely continuous at this point"
        else:
            continuity = "Likely not continuous at this point"
    else:
        continuity = "Likely not continuous or undefined at this point"

    left_derivative = np.nan
    right_derivative = np.nan

    if np.isfinite(y0) and np.isfinite(y_left):
        left_derivative = (y0 - y_left) / h

    if np.isfinite(y0) and np.isfinite(y_right):
        right_derivative = (y_right - y0) / h

    if np.isfinite(left_derivative) and np.isfinite(right_derivative):
        if abs(left_derivative - right_derivative) < 1e-2:
            differentiability = "Likely differentiable at this point"
        else:
            differentiability = "Likely not differentiable at this point"
    else:
        differentiability = "Likely not differentiable or undefined at this point"

    return {
        "y0": y0,
        "left_value": y_left,
        "right_value": y_right,
        "left_derivative": left_derivative,
        "right_derivative": right_derivative,
        "continuity": continuity,
        "differentiability": differentiability,
    }


# ============================================================
# Tabs
# ============================================================

tab_graph, tab_table, tab_complex, tab_help = st.tabs(
    [
        "Function / Inequality Graph",
        "Tables & Regression",
        "Complex Functions",
        "Help / Examples"
    ]
)


# ============================================================
# Tab 1: Function / Inequality Graph
# ============================================================

with tab_graph:
    with st.sidebar:
        st.header("Graph Controls")

        function_text = st.text_input(
            "Enter function or inequality",
            value="f(x) = x^2 {1<x<5}",
            help="Examples: f(x)=sinx, y<x^2{1<x<5}, f(x)={x<0:-x,otherwise:x}"
        )

        resolution = st.slider(
            "Graph resolution",
            min_value=100,
            max_value=1000,
            value=500,
            step=50
        )

        show_grid = st.checkbox("Show grid lines", value=True)
        show_extrema = st.checkbox("Show approximate local extrema", value=True)
        show_tangent = st.checkbox("Show tangent line", value=True)
        show_analysis = st.checkbox("Show continuity/differentiability", value=True)

    try:
        expr, variables, normalized_text, restriction_text, inequality_info = parse_function(function_text)

        st.markdown("## Parsed Input")

        if variables:
            variable_text = ", ".join(str(v) for v in variables)
            st.latex(f"f({variable_text}) = {sp.latex(expr)}")
        else:
            st.latex(sp.latex(expr))

        with st.expander("Debug: normalized expression and restriction"):
            st.write("Expression:")
            st.code(normalized_text)
            st.write("Restriction:")
            st.code(restriction_text if restriction_text else "None")
            st.write("Inequality:")
            st.code(inequality_info if inequality_info else "None")

        if len(variables) == 1:
            var = variables[0]

            col1, col2 = st.columns(2)

            with col1:
                x_min = st.number_input(f"{var} minimum", value=-10.0, key="graph_x_min")

            with col2:
                x_max = st.number_input(f"{var} maximum", value=10.0, key="graph_x_max")

            if x_min >= x_max:
                st.error("Minimum must be smaller than maximum.")
            else:
                x_vals = np.linspace(x_min, x_max, resolution)
                raw_y = evaluate_raw_1d(expr, var, x_vals)
                y_vals = clean_real_array(raw_y)
                y_vals = apply_restrictions(x_vals, y_vals, restriction_text)

                fig = go.Figure()

                if inequality_info:
                    y_plot_min = np.nanmin(y_vals) - 5 if np.any(np.isfinite(y_vals)) else -10
                    y_plot_max = np.nanmax(y_vals) + 5 if np.any(np.isfinite(y_vals)) else 10

                    if inequality_info in ["<", "<="]:
                        fill_y = np.full_like(x_vals, y_plot_min)
                    else:
                        fill_y = np.full_like(x_vals, y_plot_max)

                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode="lines",
                            line=dict(width=3, color="#c77dff"),
                            name="Boundary"
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate([x_vals, x_vals[::-1]]),
                            y=np.concatenate([y_vals, fill_y[::-1]]),
                            fill="toself",
                            fillcolor="rgba(199, 125, 255, 0.25)",
                            line=dict(color="rgba(255,255,255,0)"),
                            name=f"Region y {inequality_info} f(x)"
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode="lines",
                            line=dict(width=4, color="#c77dff"),
                            name=f"f({var})"
                        )
                    )

                maxima, minima = find_approx_local_extrema(x_vals, y_vals)

                if show_extrema and not inequality_info:
                    if maxima:
                        fig.add_trace(
                            go.Scatter(
                                x=[p[0] for p in maxima],
                                y=[p[1] for p in maxima],
                                mode="markers",
                                marker=dict(size=10, color="#ff5fa2"),
                                name="Approx local maxima"
                            )
                        )

                    if minima:
                        fig.add_trace(
                            go.Scatter(
                                x=[p[0] for p in minima],
                                y=[p[1] for p in minima],
                                mode="markers",
                                marker=dict(size=10, color="#4cc9f0"),
                                name="Approx local minima"
                            )
                        )

                tangent_x0 = None

                if show_tangent and not inequality_info:
                    tangent_x0 = st.slider(
                        "Point for tangent/analysis",
                        min_value=float(x_min),
                        max_value=float(x_max),
                        value=float((x_min + x_max) / 2),
                        step=float((x_max - x_min) / 500)
                    )

                    tangent_y, y0, slope, derivative_expr = tangent_line_values(
                        expr,
                        var,
                        tangent_x0,
                        x_vals
                    )

                    if tangent_y is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=x_vals,
                                y=tangent_y,
                                mode="lines",
                                line=dict(width=2, dash="dash", color="#4cc9f0"),
                                name="Tangent line"
                            )
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=[tangent_x0],
                                y=[y0],
                                mode="markers",
                                marker=dict(size=12, color="#ffffff"),
                                name="Tangent point"
                            )
                        )

                fig.update_layout(
                    title="Graph",
                    xaxis_title=str(var),
                    yaxis_title="y",
                    height=650,
                    template="plotly_dark",
                    hovermode="x unified",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                )

                fig.update_xaxes(showgrid=show_grid, zeroline=True)
                fig.update_yaxes(showgrid=show_grid, zeroline=True)

                st.plotly_chart(fig, use_container_width=True)

                if not inequality_info:
                    st.markdown("## Calculus Characteristics")

                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.markdown("### Approximate Local Extrema")

                        if maxima:
                            st.markdown("**Local maxima:**")
                            for x_peak, y_peak in maxima[:10]:
                                st.write(f"x ≈ {x_peak:.5g}, f(x) ≈ {y_peak:.5g}")
                        else:
                            st.write("No approximate local maxima detected.")

                        if minima:
                            st.markdown("**Local minima:**")
                            for x_low, y_low in minima[:10]:
                                st.write(f"x ≈ {x_low:.5g}, f(x) ≈ {y_low:.5g}")
                        else:
                            st.write("No approximate local minima detected.")

                    with col_b:
                        if show_tangent and tangent_x0 is not None:
                            y0 = evaluate_expression_scalar(expr, var, tangent_x0)
                            slope, derivative_expr = approximate_derivative_at(expr, var, tangent_x0)

                            st.markdown("### Tangent Line")

                            if np.isfinite(y0) and np.isfinite(slope):
                                st.write(f"At x = {tangent_x0:.5g}:")
                                st.write(f"f(x) ≈ {y0:.5g}")
                                st.write(f"f'(x) ≈ {slope:.5g}")
                                st.latex(f"y - {y0:.5g} = {slope:.5g}(x - {tangent_x0:.5g})")
                            else:
                                st.write("Could not compute tangent line at this point.")

                    if show_analysis and tangent_x0 is not None:
                        st.markdown("### Continuity and Differentiability Check")

                        analysis = analyze_continuity_and_differentiability(
                            expr,
                            var,
                            tangent_x0
                        )

                        st.write(f"Point checked: x = {tangent_x0:.5g}")
                        st.write(f"Function value: {analysis['y0']}")
                        st.write(f"Left nearby value: {analysis['left_value']}")
                        st.write(f"Right nearby value: {analysis['right_value']}")
                        st.write(f"Left derivative estimate: {analysis['left_derivative']}")
                        st.write(f"Right derivative estimate: {analysis['right_derivative']}")

                        st.success(analysis["continuity"])
                        st.info(analysis["differentiability"])

                    with st.expander("Symbolic Information"):
                        try:
                            derivative = sp.diff(expr, var)
                            st.markdown("### Derivative")
                            st.latex(sp.latex(derivative))
                        except Exception:
                            st.write("Derivative could not be calculated.")

                        try:
                            integral = sp.integrate(expr, var)
                            st.markdown("### Indefinite Integral")
                            st.latex(sp.latex(integral))
                        except Exception:
                            st.write("Integral could not be calculated.")

        elif len(variables) >= 2:
            st.markdown("## 3D Surface Graph")

            color_choice = st.selectbox(
                "Surface color theme",
                list(CUSTOM_COLORSCALES.keys()),
                index=0
            )

            show_colorbar = st.checkbox("Show color scale bar", value=True, key="surface_colorbar")

            col1, col2 = st.columns(2)

            with col1:
                x_axis_var = st.selectbox("X-axis variable", variables, index=0, format_func=str)

            with col2:
                y_axis_var = st.selectbox("Y-axis variable", variables, index=1, format_func=str)

            col3, col4 = st.columns(2)

            with col3:
                x_min = st.number_input(f"{x_axis_var} minimum", value=-5.0)
                y_min = st.number_input(f"{y_axis_var} minimum", value=-5.0)

            with col4:
                x_max = st.number_input(f"{x_axis_var} maximum", value=5.0)
                y_max = st.number_input(f"{y_axis_var} maximum", value=5.0)

            fixed_values = {}

            extra_vars = [
                var for var in variables
                if var not in [x_axis_var, y_axis_var]
            ]

            if extra_vars:
                st.markdown("### Extra Variable Controls")

                for var in extra_vars:
                    fixed_values[var] = st.slider(
                        f"{var}",
                        min_value=-10.0,
                        max_value=10.0,
                        value=1.0,
                        step=0.1
                    )

            if x_min >= x_max or y_min >= y_max:
                st.error("Minimum values must be smaller than maximum values.")
            else:
                x_grid = np.linspace(x_min, x_max, resolution)
                y_grid = np.linspace(y_min, y_max, resolution)

                X, Y = np.meshgrid(x_grid, y_grid)

                substituted_expr = expr.subs(fixed_values)
                Z = evaluate_expression_2d(substituted_expr, x_axis_var, y_axis_var, X, Y)

                fig = go.Figure(
                    data=[
                        go.Surface(
                            x=X,
                            y=Y,
                            z=Z,
                            colorscale=CUSTOM_COLORSCALES[color_choice],
                            showscale=show_colorbar,
                            lighting=dict(
                                ambient=0.45,
                                diffuse=0.8,
                                specular=0.4,
                                roughness=0.5,
                                fresnel=0.2
                            ),
                            contours={
                                "z": {
                                    "show": True,
                                    "usecolormap": True,
                                    "highlightcolor": "#ffffff",
                                    "project_z": True
                                }
                            }
                        )
                    ]
                )

                fig.update_layout(
                    title=f"Surface Plot of f({x_axis_var}, {y_axis_var})",
                    template="plotly_dark",
                    height=780,
                    paper_bgcolor="#0e1117",
                    scene=dict(
                        xaxis=dict(
                            title=str(x_axis_var),
                            showgrid=show_grid,
                            zeroline=True,
                            backgroundcolor="#0e1117"
                        ),
                        yaxis=dict(
                            title=str(y_axis_var),
                            showgrid=show_grid,
                            zeroline=True,
                            backgroundcolor="#0e1117"
                        ),
                        zaxis=dict(
                            title="f",
                            showgrid=show_grid,
                            zeroline=True,
                            backgroundcolor="#0e1117"
                        ),
                        camera=dict(
                            eye=dict(x=1.6, y=1.6, z=1.1)
                        )
                    ),
                    margin=dict(l=0, r=0, t=60, b=0)
                )

                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Symbolic Information"):
                    st.markdown("### Partial Derivatives")

                    for var in variables:
                        st.markdown(f"With respect to `{var}`:")
                        try:
                            st.latex(sp.latex(sp.diff(expr, var)))
                        except Exception:
                            st.write("Could not calculate this partial derivative.")

    except Exception as e:
        st.error("Could not parse or graph this input.")
        st.exception(e)


# ============================================================
# Tab 2: Tables and Regression
# ============================================================

with tab_table:
    st.markdown("## Tables, Data Sets, and Lines of Best Fit")

    uploaded_file = st.file_uploader("Import CSV file", type=["csv"])

    if uploaded_file is not None:
        table_df = pd.read_csv(uploaded_file)
    else:
        table_df = pd.DataFrame(
            {
                "x": [0, 1, 2, 3, 4, 5],
                "y": [1, 2.1, 3.9, 6.2, 8.1, 10.2],
            }
        )

    edited_df = st.data_editor(
        table_df,
        num_rows="dynamic",
        use_container_width=True
    )

    numeric_columns = edited_df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_columns) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            x_col = st.selectbox("X column", numeric_columns, index=0)

        with col2:
            y_col = st.selectbox("Y column", numeric_columns, index=1)

        clean_df = edited_df[[x_col, y_col]].dropna()

        x_data = clean_df[x_col].to_numpy(dtype=float)
        y_data = clean_df[y_col].to_numpy(dtype=float)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="markers",
                marker=dict(size=9, color="#ff5fa2"),
                name="Data points"
            )
        )

        regression_type = st.selectbox(
            "Regression type",
            ["None", "Linear", "Quadratic", "Polynomial"]
        )

        if regression_type != "None" and len(x_data) >= 2:
            if regression_type == "Linear":
                degree = 1
            elif regression_type == "Quadratic":
                degree = 2
            else:
                degree = st.slider("Polynomial degree", 1, 8, 3)

            if len(x_data) > degree:
                coeffs = np.polyfit(x_data, y_data, degree)
                poly = np.poly1d(coeffs)

                x_fit = np.linspace(np.min(x_data), np.max(x_data), 500)
                y_fit = poly(x_fit)

                fig.add_trace(
                    go.Scatter(
                        x=x_fit,
                        y=y_fit,
                        mode="lines",
                        line=dict(width=4, color="#4cc9f0"),
                        name=f"Best fit degree {degree}"
                    )
                )

                y_pred = poly(x_data)
                ss_res = np.sum((y_data - y_pred) ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

                equation_terms = []

                for power, coefficient in zip(range(degree, -1, -1), coeffs):
                    if power == 0:
                        equation_terms.append(f"{coefficient:.5g}")
                    elif power == 1:
                        equation_terms.append(f"{coefficient:.5g}x")
                    else:
                        equation_terms.append(f"{coefficient:.5g}x^{power}")

                equation_text = " + ".join(equation_terms).replace("+ -", "- ")

                st.markdown("### Regression Result")
                st.latex(f"y \\approx {equation_text}")
                st.write(f"R² ≈ {r_squared:.5g}")

        fig.update_layout(
            title="Data Table Plot",
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=650,
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Your table needs at least two numeric columns.")


# ============================================================
# Tab 3: Complex Functions
# ============================================================

with tab_complex:
    st.markdown("## Complex-Valued Function Viewer")

    complex_input = st.text_input(
        "Enter complex-valued function",
        value="f(x) = exp(i*x)",
        help="Examples: f(x)=exp(i*x), f(x)=sin(x)+i*cos(x)"
    )

    view_mode = st.selectbox(
        "Visualize",
        ["Real part", "Imaginary part", "Magnitude", "Phase / Argument"]
    )

    col1, col2 = st.columns(2)

    with col1:
        complex_x_min = st.number_input("x minimum", value=-10.0, key="complex_x_min")

    with col2:
        complex_x_max = st.number_input("x maximum", value=10.0, key="complex_x_max")

    try:
        expr, variables, normalized_text, restriction_text, inequality_info = parse_function(complex_input)

        if len(variables) != 1:
            st.warning("Complex viewer currently supports one-variable functions only.")
        else:
            var = variables[0]
            x_vals = np.linspace(complex_x_min, complex_x_max, 1000)
            raw_values = evaluate_raw_1d(expr, var, x_vals)
            y_view = complex_to_view(raw_values, view_mode)

            y_view = np.array(y_view, dtype=float)
            y_view[~np.isfinite(y_view)] = np.nan

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_view,
                    mode="lines",
                    line=dict(width=4, color="#c77dff"),
                    name=view_mode
                )
            )

            fig.update_layout(
                title=f"{view_mode} of Complex Function",
                xaxis_title=str(var),
                yaxis_title=view_mode,
                height=650,
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
            )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Normalized expression"):
                st.code(normalized_text)

    except Exception as e:
        st.error("Could not parse complex function.")
        st.exception(e)


# ============================================================
# Tab 4: Help
# ============================================================

with tab_help:
    st.markdown("## Supported Syntax Examples")

    st.markdown("### Basic Functions")
    st.code("f(x) = sin(x)")
    st.code("f(x) = sinx")
    st.code("f(x) = x^2 + 3x + 1")
    st.code("f(x) = sqrt(x)")
    st.code("f(x) = abs(x)")
    st.code("f(x) = |x|")

    st.markdown("### Trig and Advanced Functions")
    st.code("f(x) = sin^2(x) + cos^2(x)")
    st.code("f(x) = arcsin(x)")
    st.code("f(x) = sinh(x)")
    st.code("f(x) = floor(x)")
    st.code("f(x) = ceil(x)")
    st.code("f(x) = x! / x^x")

    st.markdown("### Piecewise Functions")
    st.code("f(x) = {x<0:-x, otherwise:x}")
    st.code("f(x) = {x<-1:-1, x<=1:x, otherwise:1}")
    st.code("f(x) = Piecewise((x^2,x<0),(sin(x),x>=0))")

    st.markdown("### Restrictions")
    st.code("f(x) = x^2 {1<x<5}")
    st.code("f(x) = sin(x) {x>0}")
    st.code("f(x) = x^2 {-2<x<2, y<3}")

    st.markdown("### Inequalities")
    st.code("y < x^2")
    st.code("y <= x^2 {1<x<5}")
    st.code("y > sin(x)")

    st.markdown("### Tables and Regression")
    st.write("Use the Tables & Regression tab to create or import data sets.")
    st.write("Supported regressions: linear, quadratic, and polynomial.")

    st.markdown("### Complex Functions")
    st.code("f(x) = exp(i*x)")
    st.code("f(x) = sin(x) + i*cos(x)")

    st.markdown("### Available 3D Surface Themes")
    for theme_name in CUSTOM_COLORSCALES.keys():
        st.write(f"- {theme_name}")
