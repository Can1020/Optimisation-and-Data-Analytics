import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional

# Tolerance constants for numerical comparisons
TOLERANCE_PARALLEL = 1e-9
TOLERANCE_CONSTRAINT = 1e-4
MAX_CONSTRAINT_VALUE = 1e5


class OptimizationType(Enum):
    """Enumeration for optimization type."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    
    def __str__(self):
        return "MAXIMIZING" if self == OptimizationType.MAXIMIZE else "MINIMIZING"


@dataclass
class Constraint:
    """Represents a linear constraint."""
    coefficients: List[float]
    rhs: float
    
    @property
    def coef1(self) -> float:
        return self.coefficients[0]
    
    @property
    def coef2(self) -> float:
        return self.coefficients[1]


@dataclass
class LPProblem:
    """Represents a Linear Programming problem."""
    objective_coefficients: List[float]
    inequality_constraints: List[Constraint]
    equality_constraints: List[Constraint]
    optimization_type: OptimizationType
    
    @property
    def c(self) -> List[float]:
        return self.objective_coefficients
    
    def get_inequality_matrices(self) -> Tuple[Optional[List[List[float]]], Optional[List[float]]]:
        """Convert inequality constraints to matrix form."""
        if not self.inequality_constraints:
            return None, None
        A_ub = [c.coefficients for c in self.inequality_constraints]
        b_ub = [c.rhs for c in self.inequality_constraints]
        return A_ub, b_ub
    
    def get_equality_matrices(self) -> Tuple[Optional[List[List[float]]], Optional[List[float]]]:
        """Convert equality constraints to matrix form."""
        if not self.equality_constraints:
            return None, None
        A_eq = [c.coefficients for c in self.equality_constraints]
        b_eq = [c.rhs for c in self.equality_constraints]
        return A_eq, b_eq
    
    def is_maximization(self) -> bool:
        return self.optimization_type == OptimizationType.MAXIMIZE


@dataclass
class Solution:
    """Represents the solution to an LP problem."""
    x1: float
    x2: float
    objective_value: float
    
    def __str__(self):
        return f"Z = {self.objective_value:.2f} at (x1={self.x1:.2f}, x2={self.x2:.2f})"

def find_feasible_vertices(problem: LPProblem) -> np.ndarray:
    """
    Find all vertices of the feasible region by intersecting constraint lines.
    """
    lines = _collect_constraint_lines(problem)
    intersection_points = _find_line_intersections(lines)
    feasible_points = _filter_feasible_points(intersection_points, problem)
    
    return np.array(feasible_points) if feasible_points else np.array([])


def _collect_constraint_lines(problem: LPProblem) -> List[Tuple[float, float, float]]:
    """Collect all constraint lines including non-negativity constraints."""
    lines = []
    
    # Add inequality constraints
    for constraint in problem.inequality_constraints:
        lines.append((constraint.coef1, constraint.coef2, constraint.rhs))
    
    # Add equality constraints
    for constraint in problem.equality_constraints:
        lines.append((constraint.coef1, constraint.coef2, constraint.rhs))
    
    # Add non-negativity constraints: x1 >= 0, x2 >= 0
    lines.extend([(1, 0, 0), (0, 1, 0)])
    
    return lines


def _find_line_intersections(lines: List[Tuple[float, float, float]]) -> List[List[float]]:
    """Find all intersection points between pairs of lines."""
    intersections = []
    
    for i, line1 in enumerate(lines):
        for line2 in lines[i + 1:]:
            intersection = _intersect_two_lines(line1, line2)
            if intersection is not None:
                intersections.append(intersection)
    
    return intersections


def _intersect_two_lines(line1: Tuple[float, float, float], 
                         line2: Tuple[float, float, float]) -> Optional[List[float]]:
    """Find intersection point of two lines. Returns None if lines are parallel."""
    a1, b1, rhs1 = line1
    a2, b2, rhs2 = line2
    
    determinant = a1 * b2 - a2 * b1
    
    if abs(determinant) <= TOLERANCE_PARALLEL:
        return None  # Lines are parallel
    
    x = (rhs1 * b2 - rhs2 * b1) / determinant
    y = (a1 * rhs2 - a2 * rhs1) / determinant
    
    return [x, y]


def _filter_feasible_points(points: List[List[float]], problem: LPProblem) -> List[List[float]]:
    """Filter points that satisfy all constraints."""
    return [point for point in points if _is_point_feasible(point, problem)]


def _is_point_feasible(point: List[float], problem: LPProblem) -> bool:
    """Check if a point satisfies all constraints."""
    x, y = point
    
    # Check non-negativity
    if x < -TOLERANCE_CONSTRAINT or y < -TOLERANCE_CONSTRAINT:
        return False
    
    # Check inequality constraints
    for constraint in problem.inequality_constraints:
        value = constraint.coef1 * x + constraint.coef2 * y
        if value > constraint.rhs + TOLERANCE_CONSTRAINT:
            return False
    
    # Check equality constraints
    for constraint in problem.equality_constraints:
        value = constraint.coef1 * x + constraint.coef2 * y
        if abs(value - constraint.rhs) > TOLERANCE_CONSTRAINT:
            return False
    
    return True

def get_user_input() -> LPProblem:
    """Get LP problem parameters from user."""
    _print_header()
    
    optimization_type = _get_optimization_type()
    objective_coefficients = _get_objective_coefficients()
    inequality_constraints, equality_constraints = _get_constraints()
    
    return LPProblem(
        objective_coefficients=objective_coefficients,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        optimization_type=optimization_type
    )


def _print_header():
    """Print input header."""
    print("\n" + "=" * 40)
    print("LP PROBLEM INPUT")
    print("=" * 40)


def _get_optimization_type() -> OptimizationType:
    """Ask user for optimization type."""
    print("\nOptimization Type:")
    print("  1. Maximize")
    print("  2. Minimize")
    
    while True:
        mode = input("Select (1/2): ").strip()
        if mode == '1':
            return OptimizationType.MAXIMIZE
        elif mode == '2':
            return OptimizationType.MINIMIZE
        print("Invalid choice. Please enter 1 or 2.")

def _get_objective_coefficients() -> List[float]:
    """Get objective function coefficients from user."""
    print("\nObjective Function: Z = c1*x1 + c2*x2")
    
    while True:
        try:
            c1 = float(input("  c1: "))
            c2 = float(input("  c2: "))
            
            if c1 == 0 and c2 == 0:
                print("Error: Both coefficients cannot be zero.")
                continue
            
            return [c1, c2]
        except ValueError:
            print("Invalid input. Please enter numeric values.")

def _get_constraints() -> Tuple[List[Constraint], List[Constraint]]:
    """Get all constraints from user."""
    inequality_constraints = []
    equality_constraints = []
    
    print("\nConstraints:")
    print("  Format: a b op rhs")
    print("  Operators: <= (less or equal), >= (greater or equal), = (equal)")
    print("  Example: 2 3 <= 10  means  2*x1 + 3*x2 <= 10")
    print("  Type 'done' when finished.\n")
    
    count = 1
    while True:
        entry = input(f"  Constraint {count}: ").strip()
        
        if entry.lower() == 'done':
            break
        
        constraint = _parse_constraint(entry)
        if constraint is not None:
            constraint_type, constraint_obj = constraint
            if constraint_type == 'inequality':
                inequality_constraints.append(constraint_obj)
            else:
                equality_constraints.append(constraint_obj)
            count += 1
    
    if not inequality_constraints and not equality_constraints:
        print("\n⚠ Warning: No constraints entered. Problem may be unbounded.")
    
    return inequality_constraints, equality_constraints


def _parse_constraint(entry: str) -> Optional[Tuple[str, Constraint]]:
    """Parse a constraint string. Returns ('inequality'|'equality', Constraint) or None."""
    try:
        parts = entry.split()
        if len(parts) != 4:
            print("  Invalid format. Use: a b op rhs (e.g., '2 3 <= 10')")
            return None
        
        coef1 = float(parts[0])
        coef2 = float(parts[1])
        operator = parts[2]
        rhs = float(parts[3])
        
        if operator == '<=':
            return ('inequality', Constraint([coef1, coef2], rhs))
        elif operator == '>=':
            # Convert >= to <= by multiplying by -1
            return ('inequality', Constraint([-coef1, -coef2], -rhs))
        elif operator == '=':
            return ('equality', Constraint([coef1, coef2], rhs))
        else:
            print(f"  Invalid operator '{operator}'. Use: <=, >=, or =")
            return None
        
    except ValueError:
        print("  Invalid input. Use numeric values for a, b, and rhs.")
        return None

def solve_and_plot(problem: LPProblem, title: str = "LP Graphical Solution"):
    """Solve LP problem and visualize the solution."""
    _print_problem_header(problem)
    _validate_problem(problem)
    
    solution = solve_lp_problem(problem)
    if solution is None:
        return
    
    print_solution(solution)
    visualize_solution(problem, solution, title)


def _print_problem_header(problem: LPProblem):
    """Print problem information."""
    print(f"\n{'=' * 50}")
    print(f"{problem.optimization_type}: Z = {problem.c[0]}*x1 + {problem.c[1]}*x2")
    print(f"{'=' * 50}")


def _validate_problem(problem: LPProblem):
    """Validate problem has constraints."""
    if not problem.inequality_constraints and not problem.equality_constraints:
        print("⚠ Warning: No constraints provided. Problem may be unbounded.")


def solve_lp_problem(problem: LPProblem) -> Optional[Solution]:
    """Solve the LP problem using scipy.linprog."""
    try:
        result = _run_linprog_solver(problem)
        
        if not result.success:
            _print_solver_error(result.status, result.message)
            return None
        
        return _create_solution_from_result(result, problem)
        
    except Exception as e:
        print(f"\n✗ Calculation Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def _run_linprog_solver(problem: LPProblem):
    """Run scipy linprog solver."""
    # linprog minimizes by default, negate coefficients for maximization
    objective = ([-c for c in problem.c] 
                if problem.is_maximization() 
                else problem.c)
    
    bounds = [(0, None), (0, None)]  # Non-negativity
    A_ub, b_ub = problem.get_inequality_matrices()
    A_eq, b_eq = problem.get_equality_matrices()
    
    return linprog(objective, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')


def _create_solution_from_result(result, problem: LPProblem) -> Solution:
    """Create Solution object from linprog result."""
    x1, x2 = result.x
    objective_value = result.fun * (-1 if problem.is_maximization() else 1)
    
    return Solution(x1=x1, x2=x2, objective_value=objective_value)

def _print_solver_error(status: int, message: str):
    """Print appropriate error message based on solver status."""
    if status == 2:
        print("\n✗ Problem is INFEASIBLE: No solution satisfies all constraints.")
    elif status == 3:
        print("\n✗ Problem is UNBOUNDED: Objective can increase/decrease indefinitely.")
    else:
        print(f"\n✗ Solver Error: {message}")


def print_solution(solution: Solution):
    """Print solution details."""
    print(f"\n✓ Optimal Solution Found:")
    print(f"  {solution}")

def visualize_solution(problem: LPProblem, solution: Solution, title: str):
    """Create visualization of the LP problem solution."""
    try:
        plt.figure(figsize=(10, 8))
        
        plot_limit = _calculate_plot_limit(problem, solution)
        x_values = np.linspace(0, plot_limit, 400)
        
        feasible_points = find_feasible_vertices(problem)
        _plot_feasible_region(feasible_points, plot_limit)
        _plot_all_constraints(problem, x_values, plot_limit)
        _plot_objective_line(problem, solution, x_values)
        _plot_optimal_point(solution)
        _configure_plot(plot_limit, title)
        
        plt.show()
        
    except Exception as e:
        print(f"\n✗ Visualization Error: {e}")
        import traceback
        traceback.print_exc()


def _calculate_plot_limit(problem: LPProblem, solution: Solution) -> float:
    """Calculate appropriate plot limits."""
    values = [abs(solution.x1) * 1.5, abs(solution.x2) * 1.5, 10]
    
    for constraint in problem.inequality_constraints + problem.equality_constraints:
        if abs(constraint.rhs) < MAX_CONSTRAINT_VALUE:
            values.append(abs(constraint.rhs))
    
    return max(values) * 1.2
def _plot_feasible_region(feasible_points: np.ndarray, plot_limit: float):
    """Plot the feasible region."""
    if len(feasible_points) == 0:
        print("⚠ Warning: No feasible vertices found.")
        plt.text(plot_limit / 2, plot_limit / 2, 'NO FEASIBLE REGION',
                fontsize=16, ha='center', color='red', weight='bold')
        return
    
    if len(feasible_points) >= 3:
        hull = ConvexHull(feasible_points)
        vertices = feasible_points[hull.vertices]
        plt.fill(vertices[:, 0], vertices[:, 1], color='#d9f9d9',
                alpha=0.5, label='Feasible Region')
        plt.plot(np.append(vertices[:, 0], vertices[0, 0]),
                np.append(vertices[:, 1], vertices[0, 1]),
                'g-', linewidth=2)
    elif len(feasible_points) == 2:
        plt.plot(feasible_points[:, 0], feasible_points[:, 1],
                'g-', linewidth=3, label='Feasible Line', marker='o')
    else:
        plt.plot(feasible_points[0, 0], feasible_points[0, 1],
                'go', markersize=10, label='Feasible Point')


def _plot_all_constraints(problem: LPProblem, x_values: np.ndarray, plot_limit: float):
    """Plot all constraint lines."""
    colors = plt.cm.get_cmap('tab10')
    
    for idx, constraint in enumerate(problem.inequality_constraints, 1):
        _plot_constraint_line(constraint, '<=', idx, x_values, plot_limit, colors, '--')
    
    constraint_num = len(problem.inequality_constraints) + 1
    for constraint in problem.equality_constraints:
        _plot_constraint_line(constraint, '=', constraint_num, x_values, plot_limit, colors, '-')
        constraint_num += 1


def _plot_constraint_line(constraint: Constraint, operator: str, num: int,
                         x_values: np.ndarray, plot_limit: float, colors, linestyle: str):
    """Plot a single constraint line."""
    color = colors(num % 10)
    label = f'C{num}: {constraint.coef1:.1f}x1 + {constraint.coef2:.1f}x2 {operator} {constraint.rhs:.1f}'
    linewidth = 2 if operator == '=' else 1.5
    
    if abs(constraint.coef2) > TOLERANCE_PARALLEL:
        y_values = (constraint.rhs - constraint.coef1 * x_values) / constraint.coef2
        plt.plot(x_values, y_values, label=label, color=color,
                linestyle=linestyle, linewidth=linewidth, alpha=0.7)
    elif abs(constraint.coef1) > TOLERANCE_PARALLEL:
        x_line = constraint.rhs / constraint.coef1
        if 0 <= x_line <= plot_limit:
            plt.axvline(x=x_line, label=label, color=color,
                       linestyle=linestyle, linewidth=linewidth, alpha=0.7)


def _plot_objective_line(problem: LPProblem, solution: Solution, x_values: np.ndarray):
    """Plot the objective function line at optimal value."""
    c = problem.c
    
    if abs(c[1]) > TOLERANCE_PARALLEL:
        y_values = (solution.objective_value - c[0] * x_values) / c[1]
        plt.plot(x_values, y_values, 'r-', linewidth=3,
                label=f'Objective: Z = {solution.objective_value:.2f}', alpha=0.8)
    elif abs(c[0]) > TOLERANCE_PARALLEL:
        plt.axvline(x=solution.x1, color='r', linewidth=3,
                   label=f'Objective: Z = {solution.objective_value:.2f}', alpha=0.8)
    else:
        print("⚠ Warning: Both objective coefficients are near zero.")


def _plot_optimal_point(solution: Solution):
    """Plot the optimal solution point."""
    plt.plot(solution.x1, solution.x2, 'ro', markersize=12,
            markeredgecolor='black', markeredgewidth=2,
            label='Optimal Solution', zorder=5)
    
    plt.annotate(f'({solution.x1:.1f}, {solution.x2:.1f})\nZ = {solution.objective_value:.1f}',
                xy=(solution.x1, solution.x2), xytext=(10, 10),
                textcoords='offset points', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                fontweight='bold')


def _configure_plot(plot_limit: float, title: str):
    """Configure plot appearance and labels."""
    plt.xlim(0, plot_limit)
    plt.ylim(0, plot_limit)
    plt.xlabel('x1 (Decision Variable 1)', fontsize=12)
    plt.ylabel('x2 (Decision Variable 2)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1), fontsize=9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()


def _create_pill_production_problem() -> LPProblem:
    """Create the pill production optimization problem."""
    return LPProblem(
        objective_coefficients=[2, 1],
        inequality_constraints=[
            Constraint([3, 2], 1000),    # Ingredient: 3x1 + 2x2 <= 1000
            Constraint([-1, 0], -100),   # x1 >= 100 -> -x1 <= -100
            Constraint([3, -2], 0)       # x2 >= 0.6(x1+x2) -> 3x1 - 2x2 <= 0
        ],
        equality_constraints=[],
        optimization_type=OptimizationType.MINIMIZE
    )


def _print_pill_production_info():
    """Print information about the pill production problem."""
    print("\n--- Loading IndustryOR 'Pill Production' Case ---")
    print("Objective: Minimize Z = 2x1 + x2 (Filler Material)")
    print("Constraint 1: 3x1 + 2x2 <= 1000 (Ingredients)")
    print("Constraint 2: x1 >= 100 (Min Large Pills)")
    print("Constraint 3: x2 >= 60% of Total Pills")


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("GRAPHICAL LP SOLVER")
    print("-------------------")
    print("Choose Data Source:")
    print("1. Enter Custom Data Manually")
    print("2. Load 'IndustryOR' Real-World Test (Pill Production)")
    
    choice = input("Choice (1/2): ")
    
    if choice == '1':
        problem = get_user_input()
        solve_and_plot(problem)
        
    elif choice == '2':
        problem = _create_pill_production_problem()
        _print_pill_production_info()
        solve_and_plot(problem, "IndustryOR: Pill Production Optimization")
    else:
        print("Invalid choice.")