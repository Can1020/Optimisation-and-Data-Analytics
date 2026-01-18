import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

# --- CONFIGURATION ---
TOLERANCE_PARALLEL = 1e-9
TOLERANCE_CONSTRAINT = 1e-4
MAX_PLOT_LIMIT = 1e4

# Plotting constants
PLOT_EXTENSION_FACTOR = 1.2
PLOT_RESOLUTION = 400
DEFAULT_PLOT_MIN = 10
OBJECTIVE_LINE_WIDTH = 2.5
OPTIMAL_POINT_SIZE = 10
FEASIBLE_REGION_RESOLUTION = 500
GRID_ALPHA = 0.3
ANNOTATION_OFFSET_X = 10
ANNOTATION_OFFSET_Y = 10
ANNOTATION_BOX_PADDING = 0.3
ANNOTATION_BOX_ALPHA = 0.8

@dataclass
class Point2D:
    """Represents a point in 2D space."""
    x_coordinate: float
    y_coordinate: float
    
    def __str__(self) -> str:
        return f"({self.x_coordinate:.1f}, {self.y_coordinate:.1f})"

@dataclass
class LinearCoefficients:
    """Represents coefficients of a linear equation: c1*x + c2*y = rhs."""
    first_coefficient: float
    second_coefficient: float
    right_hand_side: float
    
    def is_vertical_line(self, tolerance: float = TOLERANCE_PARALLEL) -> bool:
        """Check if the line is vertical (second coefficient near zero)."""
        return abs(self.second_coefficient) < tolerance
    
    def calculate_y_value(self, x_value: np.ndarray) -> np.ndarray:
        """Calculate y for given x: y = (rhs - c1*x) / c2."""
        if self.is_vertical_line():
            raise ValueError("Cannot calculate y-value for vertical line")
        return (self.right_hand_side - self.first_coefficient * x_value) / self.second_coefficient
    
    def get_x_intercept(self) -> float:
        """Get x-intercept for vertical or near-vertical lines."""
        if self.first_coefficient == 0:
            return 0.0
        return self.right_hand_side / self.first_coefficient

class OptimizationType(Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"

@dataclass
class LPProblem:
    """Stores the LP definition using symbolic and numeric representations."""
    variables: List[sp.Symbol]
    optimization_type: OptimizationType
    objective_expr: sp.Expr
    constraints: List['ParsedConstraint'] = field(default_factory=list)

    @property
    def objective_coefficients(self) -> List[float]:
        """Extracts numerical coefficients for the objective function."""
        coefficients = []
        for variable in self.variables:
            coefficient_value = float(self.objective_expr.coeff(variable))
            coefficients.append(coefficient_value)
        return coefficients

    def is_maximization(self) -> bool:
        return self.optimization_type == OptimizationType.MAXIMIZE

@dataclass
class ParsedConstraint:
    """Stores a constraint derived from a symbolic expression."""
    original_text: str
    coefficients: List[float]
    right_hand_side: float                
    operator: str
    
    def to_linear_coefficients(self) -> LinearCoefficients:
        """Convert to LinearCoefficients object."""
        return LinearCoefficients(
            first_coefficient=self.coefficients[0],
            second_coefficient=self.coefficients[1],
            right_hand_side=self.right_hand_side
        )             

@dataclass
class Solution:
    """Stores the optimal result."""
    variable_values: Dict[str, float]
    objective_value: float
    status_message: str
    is_successful: bool

    def __str__(self) -> str:
        if not self.is_successful:
            return f"No Solution: {self.status_message}"
        
        variables_string = ", ".join([f"{key}={value:.2f}" for key, value in self.variable_values.items()])
        return f"Optimal Z = {self.objective_value:.2f} ({variables_string})"
    
    def get_optimal_point(self) -> Point2D:
        """Get the optimal solution as a Point2D."""
        values = list(self.variable_values.values())
        return Point2D(x_coordinate=values[0], y_coordinate=values[1])

# --- INPUT PARSING LOGIC ---

def parse_user_input() -> LPProblem:
    """Handles the interactive user input session."""
    print("\n--- LINEAR PROGRAMMING SOLVER ---")
    
    # 1. Variable Setup
    var_input = input("Enter decision variables (comma separated, e.g., 'A, B' or 'x, y'): ").strip()
    var_names = [name.strip() for name in var_input.split(',')]
    if len(var_names) != 2:
        raise ValueError("This solver strictly supports exactly 2 variables.")
    
    # Create Sympy Symbols
    symbols = [sp.Symbol(name) for name in var_names]
    symbol_map = {name: symbol for name, symbol in zip(var_names, symbols)}
    
    # 2. Optimization Goal
    print("\nOptimization Goal:")
    mode_string = input("Type 'max' to Maximize or 'min' to Minimize: ").strip().lower()
    optimization_type = OptimizationType.MAXIMIZE if mode_string.startswith('max') else OptimizationType.MINIMIZE
    
    # 3. Objective Function
    objective_string = input(f"Enter Objective Function (e.g., '30*A + 10*B'): ")
    if "=" in objective_string:
        objective_string = objective_string.split("=")[1]
    
    objective_expr = _parse_expression_safely(objective_string, symbol_map)
    
    # 4. Constraints
    constraints = []
    print("\nEnter Constraints.")
    print("Examples: '6*A + 3*B <= 40', 'B >= 3*A', 'A <= 4'.")
    print("Type 'done' when finished.")
    
    count = 1
    while True:
        entry = input(f"Constraint {count}: ").strip()
        if entry.lower() == 'done':
            break
        if not entry:
            continue
            
        try:
            constraints.append(_parse_constraint_string(entry, symbols, symbol_map))
            count += 1
        except (ValueError, KeyError, AttributeError) as error:
            print(f"Error parsing constraint: {error}. Try again.")

    return LPProblem(symbols, optimization_type, objective_expr, constraints)

def _parse_expression_safely(expression_string: str, symbol_dictionary: dict) -> sp.Expr:
    """Parses a string into a Sympy expression with implicit multiplication support."""
    transformations = standard_transformations + (implicit_multiplication_application,)
    return parse_expr(expression_string, local_dict=symbol_dictionary, transformations=transformations)

def _parse_constraint_string(constraint_text: str, variables: List[sp.Symbol], symbol_dictionary: dict) -> ParsedConstraint:
    """
    Converts a raw string like 'B >= 3A' into canonical linear form: ax + by <= c
    """
    operator = _extract_operator(constraint_text)
    left_side_string, right_side_string = _split_by_operator(constraint_text, operator)
    
    left_expression = _parse_expression_safely(left_side_string, symbol_dictionary)
    right_expression = _parse_expression_safely(right_side_string, symbol_dictionary)
    
    # Sympy expressions support subtraction but type checker doesn't recognize it
    difference_expression = sp.Add(left_expression, -right_expression)
    
    coefficients = _extract_coefficients(difference_expression, variables)
    constant_value = _extract_constant_term(difference_expression)
    
    normalized_coefficients = coefficients
    normalized_rhs = -constant_value
    
    if operator == '>=':
        normalized_coefficients = [-coefficient for coefficient in coefficients]
        normalized_rhs = constant_value
        
    return ParsedConstraint(
        original_text=constraint_text,
        coefficients=normalized_coefficients,
        right_hand_side=normalized_rhs,
        operator=operator
    )

def _extract_operator(constraint_text: str) -> str:
    """Extract the comparison operator from a constraint string."""
    if '<=' in constraint_text:
        return '<='
    elif '>=' in constraint_text:
        return '>='
    elif '<' in constraint_text:
        return '<='
    elif '>' in constraint_text:
        return '>='
    elif '=' in constraint_text:
        return '='
    else:
        raise ValueError("No comparison operator found (<=, >=, =)")

def _split_by_operator(constraint_text: str, operator: str) -> tuple[str, str]:
    """Split constraint text by its operator."""
    if operator in ['<=', '>=']:
        parts = constraint_text.split(operator)
    elif '<' in constraint_text:
        parts = constraint_text.split('<')
    elif '>' in constraint_text:
        parts = constraint_text.split('>')
    else:
        parts = constraint_text.split('=')
    return parts[0], parts[1]

def _extract_coefficients(expression: sp.Expr, variables: List[sp.Symbol]) -> List[float]:
    """Extract coefficients for each variable from the expression."""
    coefficients = []
    polynomial = sp.Poly(expression, variables)
    
    for variable in variables:
        try:
            coefficient = float(polynomial.coeff_monomial(variable))
        except (ValueError, TypeError):
            coefficient = 0.0
        coefficients.append(coefficient)
    
    return coefficients

def _extract_constant_term(expression: sp.Expr) -> float:
    """Extract the constant term from the expression."""
    coefficients_dict = expression.as_coefficients_dict()
    return float(coefficients_dict.get(1, 0))

# --- SOLVER LOGIC ---

def solve_lp(problem: LPProblem) -> Solution:
    """Solve the linear programming problem using scipy's linprog."""
    objective_coefficients = problem.objective_coefficients
    if problem.is_maximization():
        objective_coefficients = [-value for value in objective_coefficients]
        
    inequality_matrix = []
    inequality_bounds = []
    equality_matrix = []
    equality_bounds = []
    
    for constraint in problem.constraints:
        if constraint.operator == '=':
            equality_matrix.append(constraint.coefficients)
            equality_bounds.append(constraint.right_hand_side)
        else:
            inequality_matrix.append(constraint.coefficients)
            inequality_bounds.append(constraint.right_hand_side)
            
    variable_bounds = [(0, None), (0, None)]
    
    result = linprog(
        objective_coefficients,
        A_ub=inequality_matrix if inequality_matrix else None,
        b_ub=inequality_bounds if inequality_bounds else None,
        A_eq=equality_matrix if equality_matrix else None,
        b_eq=equality_bounds if equality_bounds else None,
        bounds=variable_bounds,
        method='highs'
    )
    
    if result.success:
        objective_value = result.fun * (-1 if problem.is_maximization() else 1)
        variable_map = {str(variable): value for variable, value in zip(problem.variables, result.x)}
        return Solution(variable_map, objective_value, result.message, True)
    else:
        return Solution({}, 0.0, result.message, False)

# --- VISUALIZATION LOGIC ---

def visualize_result(problem: LPProblem, solution: Solution) -> None:
    """Visualize the linear programming problem and its solution."""
    if not solution.is_successful:
        print("Cannot visualize: No feasible solution found.")
        return

    plt.figure(figsize=(10, 8))
    
    plot_limit = _calculate_plot_limit(solution, problem.constraints)
    x_values = np.linspace(0, plot_limit, PLOT_RESOLUTION)
    
    _setup_axes()
    _plot_constraints(problem.constraints, x_values, plot_limit)
    _plot_feasible_region(problem.constraints, plot_limit)
    _plot_objective_function(problem, solution, x_values)
    _plot_optimal_point(problem, solution)
    _configure_plot_appearance(problem, plot_limit)
    
    plt.show()

def _calculate_plot_limit(solution: Solution, constraints: List[ParsedConstraint]) -> float:
    """Calculate appropriate plot limit based on solution and constraints."""
    values = list(solution.variable_values.values())
    max_value = max(max(values) if values else 0, DEFAULT_PLOT_MIN)
    
    for constraint in constraints:
        if constraint.right_hand_side < MAX_PLOT_LIMIT:
            max_value = max(max_value, abs(constraint.right_hand_side))
    
    return max_value * PLOT_EXTENSION_FACTOR

def _setup_axes() -> None:
    """Set up the x and y axes."""
    plt.axvline(0, color='black', linewidth=1)
    plt.axhline(0, color='black', linewidth=1)

def _plot_constraints(constraints: List[ParsedConstraint], x_values: np.ndarray, plot_limit: float) -> None:
    """Plot all constraint lines."""
    color_map = plt.cm.get_cmap('tab10')
    
    for index, constraint in enumerate(constraints):
        color = color_map(index % 10)
        coefficients = constraint.to_linear_coefficients()
        
        if coefficients.is_vertical_line():
            _plot_vertical_constraint(coefficients, constraint, color, plot_limit)
        else:
            _plot_standard_constraint(coefficients, constraint, x_values, color, plot_limit)

def _plot_vertical_constraint(coefficients: LinearCoefficients, constraint: ParsedConstraint, color: Any, plot_limit: float) -> None:
    """Plot a vertical line constraint."""
    x_intercept = coefficients.get_x_intercept()
    plt.axvline(x_intercept, color=color, linestyle='--', label=constraint.original_text)

def _plot_standard_constraint(coefficients: LinearCoefficients, constraint: ParsedConstraint, 
                              x_values: np.ndarray, color: Any, plot_limit: float) -> None:
    """Plot a standard (non-vertical) constraint."""
    y_values = coefficients.calculate_y_value(x_values)
    plt.plot(x_values, y_values, color=color, linestyle='--', label=constraint.original_text)

def _plot_feasible_region(constraints: List[ParsedConstraint], plot_limit: float) -> None:
    """Plot the feasible region where all constraints are satisfied."""
    # Create a mesh grid
    x_mesh = np.linspace(0, plot_limit, FEASIBLE_REGION_RESOLUTION)
    y_mesh = np.linspace(0, plot_limit, FEASIBLE_REGION_RESOLUTION)
    X, Y = np.meshgrid(x_mesh, y_mesh)
    
    # Initialize with all points feasible (non-negativity constraints satisfied)
    feasible_region = np.ones_like(X, dtype=bool)
    
    # Check each constraint
    for constraint in constraints:
        first_coefficient = constraint.coefficients[0]
        second_coefficient = constraint.coefficients[1]
        right_hand_side = constraint.right_hand_side
        
        # Calculate constraint satisfaction: first_coeff*x + second_coeff*y <= rhs
        constraint_satisfied = (first_coefficient * X + second_coefficient * Y) <= right_hand_side
        
        # Intersection with existing feasible region
        feasible_region = feasible_region & constraint_satisfied
    
    # Plot the feasible region with transparency
    plt.contourf(X, Y, feasible_region.astype(float), levels=[0.5, 1.5], colors=['green'], alpha=0.3)

def _plot_objective_function(problem: LPProblem, solution: Solution, x_values: np.ndarray) -> None:
    """Plot the objective function line at the optimal value."""
    objective_coefficients = problem.objective_coefficients
    optimal_value = solution.objective_value
    
    if abs(objective_coefficients[1]) > TOLERANCE_PARALLEL:
        y_objective = (optimal_value - objective_coefficients[0] * x_values) / objective_coefficients[1]
        plt.plot(x_values, y_objective, 'r-', linewidth=OBJECTIVE_LINE_WIDTH, 
                label=f'Objective Z={optimal_value:.1f}')

def _plot_optimal_point(problem: LPProblem, solution: Solution) -> None:
    """Plot and annotate the optimal solution point."""
    optimal_point = solution.get_optimal_point()
    
    plt.plot(optimal_point.x_coordinate, optimal_point.y_coordinate, 'ro', 
            markersize=OPTIMAL_POINT_SIZE, zorder=5, label='Optimal Solution')
    
    plt.annotate(
        str(optimal_point),
        (optimal_point.x_coordinate, optimal_point.y_coordinate),
        xytext=(ANNOTATION_OFFSET_X, ANNOTATION_OFFSET_Y),
        textcoords='offset points',
        bbox=dict(boxstyle=f"round,pad={ANNOTATION_BOX_PADDING}", 
                 fc="white", ec="black", alpha=ANNOTATION_BOX_ALPHA)
    )

def _configure_plot_appearance(problem: LPProblem, plot_limit: float) -> None:
    """Configure the plot's labels, limits, and styling."""
    plt.xlim(0, plot_limit)
    plt.ylim(0, plot_limit)
    plt.xlabel(str(problem.variables[0]))
    plt.ylabel(str(problem.variables[1]))
    plt.title(f"LP Optimization: {problem.optimization_type.name}")
    plt.legend()
    plt.grid(True, alpha=GRID_ALPHA)

# --- MAIN ---
def main() -> None:
    """Main entry point for the linear programming solver."""
    try:
        problem_definition = parse_user_input()
        solution = solve_lp(problem_definition)
        print("\n" + str(solution))
        visualize_result(problem_definition, solution)
    except (ValueError, KeyError, AttributeError) as error:
        print(f"\nAn error occurred: {error}")

if __name__ == "__main__":
    main()