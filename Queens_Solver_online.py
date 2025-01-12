"""
Online script: Fully automatic solver 
This is the complete script. Therefore it will log you into LinkedIn and solve today's grid. 
If you are just interested in the logic, better take a look at the offline script on my GitHub first. 
This online version here as no print statements (for efficiency), see the offline version to follow along the output.

Ready to go?
1. Clicking Run will open a chrome page, and ask you to log in (you can make this process automatic if you put your log-in credential in the code. Otherwise just do it manually)
2. Click "Start Game" button
3. See the puzzle being solved in real time

"""

# To fully automate the task, you can put your log in here. Otherwise just type them manually on the webpage when asked (in this case you might want to increase waiting time of the Macro in the Main function at the bottom)
linkedin_email = "email.example@gmail.com"
linkedin_password = "password123"

from bs4 import BeautifulSoup #for HTML parsing (decoding the grid structure from the website)
from selenium import webdriver # Using Selenium to interact with the webbrowser (get HTML, click cells, etc.)
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from time import time 
from copy import deepcopy
from itertools import combinations 


def initialize_permanent_values(temp_grid, grid_colors):
    """
    First, we analyze the initial grid and color structure to determine permanent constraints:
    - Place permanent queens (1s) where mandatory.
    - Mark permanent zeros (0s) positions where queens cannot be placed.
    - In most cases this will not fully solve the board but will narrow down possibilities for higher algorithmic efficiency later on

    Args:
    - temp_grid: 2D list representing the current state of the grid (None for unfilled cells).
    - grid_colors: 2D list indicating the color of each cell.

    Returns:
    - Updated temp_grid with PERMANENT constraints that apply (initial fixing of what we know)
    - A set of permanent_ones (fixed queens' positions).
    - A set of permanent_zeros (positions where queens are disallowed).
    """

    rows, cols = len(temp_grid), len(temp_grid[0])
    permanent_ones = set()
    permanent_zeros = set()

    # Create a loop so we re-iterate whenever any placement has been done (because blocking cells as 0, might help to block further cells in next step)
    updated = True
    while updated:
        updated = False

        # Iterate over each row to apply rules
        for r in range(rows):
            # Extract the values in the current row
            row_values = [temp_grid[r][c] for c in range(cols)]
            
            # Check if any cell in the row is already filled with a queen (1)
            if row_values.count(1) == cols: 
                # Iterate through all cells in the grid if there is a queen
                for x in range(rows):
                    for y in range(cols):
                        # Block all cells that share the same color as the current row's queen's color
                        if grid_colors[x][y] == grid_colors[r][0]:
                            temp_grid[x][y] = 0  # Mark the cell as invalid for placing queens (like setting a cross online)
                            permanent_zeros.add((x, y))  # Add to the set of permanent 0s
                            updated = True  # Indicate that the grid has been updated

        # Iterate over each column to apply similar rules
        for c in range(cols):
            col_values = [temp_grid[r][c] for r in range(rows)]
            if col_values.count(1) == rows:  
                for x in range(rows):
                    for y in range(cols):
                        if grid_colors[x][y] == grid_colors[0][c]:
                            temp_grid[x][y] = 0  
                            permanent_zeros.add((x, y))  
                            updated = True  

        # Apply rules for colors 
        unique_colors = set(color for row in grid_colors for color in row)  # Get all unique colors in the grid
        for color in unique_colors:
            # Find all cells that have the current color
            color_cells = [(r, c) for r in range(rows) for c in range(cols) if grid_colors[r][c] == color]
            
            # If there's only one cell of this color, it must be a queen
            if len(color_cells) == 1:
                r, c = color_cells[0]
                if temp_grid[r][c] is None:  # If the cell is unfilled
                    temp_grid[r][c] = 1  # Place a queen
                    permanent_ones.add((r, c))  # Add to the set of permanent queens
                    updated = True  
            
            # If this color already has a queen, block all other cells of this color
            elif all(temp_grid[r][c] == 1 for r, c in color_cells):
                for r, c in color_cells:
                    if temp_grid[r][c] is None: 
                        temp_grid[r][c] = 0 
                        permanent_zeros.add((r, c))  
                        updated = True  

        # Check rows for exclusive color dominance
        for r in range(rows):
            if any(temp_grid[r][c] == 1 for c in range(cols)):  # Skip rows with a queen
                continue

            row_colors = [grid_colors[r][c] for c in range(cols) if temp_grid[r][c] is None]
            unique_colors = set(row_colors)
            if len(unique_colors) == 1 and row_colors:  # All blanks in the row are of one color
                exclusive_color = unique_colors.pop()
                for x in range(rows):
                    for y in range(cols):
                        if grid_colors[x][y] == exclusive_color and x != r and temp_grid[x][y] is None:
                            temp_grid[x][y] = 0
                            permanent_zeros.add((x, y))
                            updated = True

                # Block the directly above and below cells if dominance is done with 2 cells (they would block these 2 cells and we know a queens MUST be there)
                blank_cells = [(r, c) for c in range(cols) if temp_grid[r][c] is None and grid_colors[r][c] == exclusive_color]
                if all(blank_cells[i][1] + 1 == blank_cells[i + 1][1] for i in range(len(blank_cells) - 1)): # checks if the blanks are next to each other
                    if len(blank_cells) == 2:  # If exactly 2 cells are blank (and next to each other)
                        for cell in blank_cells:
                            r_cell, c_cell = cell
                            for dr in [-1, 1]:
                                nr = r_cell + dr
                                if 0 <= nr < rows:
                                    temp_grid[nr][c_cell] = 0
                                    permanent_zeros.add((nr, c_cell))
                    elif len(blank_cells) == 3:  # If exactly 3 cells are blank
                        middle = blank_cells[1]  # The middle cell
                        middle_row, middle_col = middle
                        for dr in [-1, 1]:
                            nr = middle_row + dr
                            if 0 <= nr < rows:
                                temp_grid[nr][middle_col] = 0
                                permanent_zeros.add((nr, middle_col))

        # Color Dominance for cols
        for c in range(cols):
            if any(temp_grid[r][c] == 1 for r in range(rows)):  # Skip columns with a queen
                continue

            col_colors = [grid_colors[r][c] for r in range(rows) if temp_grid[r][c] is None]
            unique_colors = set(col_colors)
            if len(unique_colors) == 1 and col_colors: 
                exclusive_color = unique_colors.pop()
                for x in range(rows):
                    for y in range(cols):
                        if grid_colors[x][y] == exclusive_color and y != c and temp_grid[x][y] is None:
                            temp_grid[x][y] = 0
                            permanent_zeros.add((x, y))
                            updated = True

                # Same logic just that for colums here we block the left and right cells (instead of above and below)
                blank_cells = [(r, c) for r in range(rows) if temp_grid[r][c] is None and grid_colors[r][c] == exclusive_color]
                if all(blank_cells[i][0] + 1 == blank_cells[i + 1][0] for i in range(len(blank_cells) - 1)):
                    if len(blank_cells) == 2:  
                        for cell in blank_cells:
                            r_cell, c_cell = cell
                            for dc in [-1, 1]:
                                nc = c_cell + dc
                                if 0 <= nc < cols:
                                    temp_grid[r_cell][nc] = 0
                                    permanent_zeros.add((r_cell, nc))
                    elif len(blank_cells) == 3:  
                        middle = blank_cells[1]  # The middle cell
                        middle_row, middle_col = middle
                        for dc in [-1, 1]:
                            nc = middle_col + dc
                            if 0 <= nc < cols:
                                temp_grid[middle_row][nc] = 0
                                permanent_zeros.add((middle_row, nc))

        # Check for color confinements (if all blank cells of a color are within a row/ column, the queen for this r/c MUST be of that color = block all other colors in this r/c)
        # This applies to all adject r/c (if 2 colors are confined within 2 rows, we can block all other colors in these 2 rows)
    
        def get_adjacent_groups(size): 
            """Generate all valid groups of adjacent rows or columns. I.e: 2,3,4 but not 2,4 (not adjecent)"""
            return [list(range(start, start + length))
                    for length in range(1, size + 1)
                    for start in range(size - length + 1)]

        def is_fully_confined_row(color, row_group):
            """Check if a color is fully confined within the given rows."""
            confined_positions = []

            for r in row_group:
                for c in range(cols):
                    if grid_colors[r][c] == color and temp_grid[r][c] is None:
                        confined_positions.append((r, c))

            # Ensure all cells of this color are within the specified rows
            for r in range(rows):
                for c in range(cols):
                    if grid_colors[r][c] == color and temp_grid[r][c] is None and (r, c) not in confined_positions:
                        return False
            return True

        def is_fully_confined_col(color, col_group): # same for cols
            """Check if a color is fully confined within the given columns."""
            confined_positions = []

            for c in col_group:
                for r in range(rows):
                    if grid_colors[r][c] == color and temp_grid[r][c] is None:
                        confined_positions.append((r, c))

            for c in range(cols):
                for r in range(rows):
                    if grid_colors[r][c] == color and temp_grid[r][c] is None and (r, c) not in confined_positions:
                        return False
            return True

        def process_row_groups(groups, updated):
            for group in groups:

                # Skip row groups with queens
                if any(any(temp_grid[r][c] == 1 for c in range(cols)) for r in group):
                    continue

                # Count fully confined colors in the group
                unique_colors = set()
                for r in group:
                    for c in range(cols):
                        if temp_grid[r][c] is None:
                            unique_colors.add(grid_colors[r][c])

                fully_confined_colors = [
                    color for color in unique_colors
                    if is_fully_confined_row(color, group)
                ]

                # If number of fully confined colors matches group size (only then are fully confined) set all OTHER colors to 0
                if len(fully_confined_colors) == len(group):
                    for r in group:
                        for c in range(cols):
                            if grid_colors[r][c] not in fully_confined_colors and temp_grid[r][c] is None:
                                temp_grid[r][c] = 0
                                permanent_zeros.add((r, c))
                                updated = True

            return updated # to track if there was any update

        def process_col_groups(groups,updated): # same for cols
            for group in groups:
                if any(any(temp_grid[r][c] == 1 for r in range(rows)) for c in group):
                    continue

                unique_colors = set()
                for c in group:
                    for r in range(rows):
                        if temp_grid[r][c] is None:
                            unique_colors.add(grid_colors[r][c])

                fully_confined_colors = [
                    color for color in unique_colors
                    if is_fully_confined_col(color, group)
                ]

                if len(fully_confined_colors) == len(group):
                    for c in group:
                        for r in range(rows):
                            if grid_colors[r][c] not in fully_confined_colors and temp_grid[r][c] is None:
                                temp_grid[r][c] = 0
                                permanent_zeros.add((r, c))
                                updated = True

            return updated

        # Adjust corner handling for rows and columns (most left row can't have a left group partner)
        def process_corner_groups(size, is_row, updated):
            corner_groups = []
            for i in range(size):
                if i == 0:  # First row/column
                    corner_groups.append([i, i + 1])
                elif i == size - 1:  # Last row/column
                    corner_groups.append([i - 1, i])
                else:  # Middle rows/columns
                    corner_groups.append([i - 1, i, i + 1])

            if is_row:
                updated = process_row_groups(corner_groups, updated)
            else:
                updated = process_col_groups(corner_groups, updated)
            return updated

        # Process row and column groups
        row_groups = get_adjacent_groups(rows)
        updated = process_row_groups(row_groups, updated)
        updated = process_corner_groups(rows, is_row=True, updated=updated)

        col_groups = get_adjacent_groups(cols)
        updated = process_col_groups(col_groups, updated)
        updated = process_corner_groups(cols, is_row=False, updated=updated)

    # Further constraint propagation (set 0s) for placed queens
    for r in range(rows):
        for c in range(cols):
            if temp_grid[r][c] == 1:  # propagate constraints for placed 1s
                propagate_constraints(temp_grid, r, c, None, grid_colors, permanent_ones, permanent_zeros) #function to set 0s

    return temp_grid, permanent_ones, permanent_zeros

# Algorithm
def solve_with_backtracking_and_propagation(temp_grid, grid_colors, variables, permanent_ones=None, permanent_zeros=None):
    """
    Solves the puzzle using a combination of optimized backtracking and constraint propagation
    - initializes the grid as above, if this does not fully solve the board, we use the following backtracking algorithm:
        - places a temporary queen at the most constrained cell (least available blanks in rows, coloumns or colors)
        - thereby we can find the most critical points. So even if the placement is wrong, it will simplfy the grid much
        - iteratively checks constraints and continues to place temp queens until either the solution is found or an invalid state occurs
        - if invalid state: Backtrack only the last placed temporary queen and try all other alternatives in the current state
        - should all options of this state be exhausted, backtrack second-to-last temp queen to explore new sub-path
        - if all options in one main-path are fully explored, place a permanent 0 at the coordinates of the first placed temp queen
        - explore a new path. Since this is a deterministic puzzle, this will find the correct solution in milliseconds

    New Args:
    - r, c: Row and column indices of the placed queen.
    - variables: List of variables for constraints (unused in this implementation).
    - temporary: Boolean indicating if the propagation is temporary (can be undone).

    Returns:
    - The solved grid
    """
    
    # initialize permanent constraints
    if permanent_ones is None:
        permanent_ones = set()
    if permanent_zeros is None:
        permanent_zeros = set()

    # backtracking stacks to exclude options if they've already been tried (otherwise we'd be stuck in an infite loop, retrying the same option again)
    rows, cols = len(temp_grid), len(temp_grid[0])
    backtracking_stack = []  # Stack to track placed temporary queens and alternative options 
    first_temp_queen = None  # Track the first temporary queen in the current main-path
    tried_first_queen = set()  # Keep track of all first queens already tried
    tried_temp_queens = {}   # Map board states to queens that have already been tried

    # Step 1: Initialize permanent values based on the rules of the game (the function from above)
    temp_grid, permanent_ones, permanent_zeros = initialize_permanent_values(temp_grid, grid_colors)

    # Simplify the grid if possible: Give the final fixing of permanent values.
    temp_grid = place_permanent_queens(temp_grid, grid_colors, variables)

    # start Algo
    while True:
        # Generate a unique representation of the current board state
        board_state = tuple(tuple(row) for row in temp_grid)

        # Step 2: Validate the current grid state
        if not is_valid_state(temp_grid, grid_colors):
            # If the state is invalid, backtrack to a previous decision point
            if not backtracking_stack:
                if first_temp_queen:
                    # Mark the first queen placement of main-path as invalid (permanent zero)
                    r, c = first_temp_queen
                    temp_grid[r][c] = 0
                    permanent_zeros.add((r, c)) # at coordinated of first queen
                    tried_first_queen.add(first_temp_queen)
                    first_temp_queen = None # free-up variable so we can re-assign it to new main-path
                else:
                    return None # should not happen ;) 

            # Backtrack: Remove the last placed queen
            last_queen, tried_options, temp_zeros = backtracking_stack.pop()
            r, c = last_queen
            temp_grid[r][c] = None

            # Reset temporary zeros that were placed as a result of this temp queen
            for tr, tc in temp_zeros:
                temp_grid[tr][tc] = None

            # Update the which queens we tried for the current board state (so we dont try the same option twice)
            tried_temp_queens[board_state] = tried_temp_queens.get(board_state, set())
            tried_temp_queens[board_state].add((r, c))

            # If we exhausted all options
            if first_temp_queen and (r, c) == first_temp_queen:
                temp_grid[r][c] = 0
                permanent_zeros.add((r, c))
                tried_first_queen.add(first_temp_queen)
                first_temp_queen = None
            continue

        # Step 3: Check if solution is complete
        if is_goal_state(temp_grid):
            return temp_grid

        # Step 4: Find the most constrained cell to prioritise it
        most_constrained_cells = find_most_constrained_cells(temp_grid, grid_colors)
        next_cell = None

        # Check the most constrained cells for an untried option (exclude options we already tried)
        for cell in most_constrained_cells:
            if first_temp_queen is None and cell in tried_first_queen:
                continue

            if cell not in tried_temp_queens.get(board_state, set()) and not any(cell == queen for queen, _, _ in backtracking_stack):
                next_cell = cell
                break

        if not next_cell: # if there is no option left, we remove the last placed temp queen to try alternatives
            print("No valid cell to place a queen. Backtracking...")
            if not backtracking_stack:
                return None

            # Backtrack: Remove last placed queen
            last_queen, tried_options, temp_zeros = backtracking_stack.pop()
            r, c = last_queen
            temp_grid[r][c] = None

            # Reset temporary zeros placed as a result of this queen
            for tr, tc in temp_zeros:
                temp_grid[tr][tc] = None

            # Update
            tried_temp_queens[board_state] = tried_temp_queens.get(board_state, set())
            tried_temp_queens[board_state].add((r, c))

            continue

        # Step 5: Place a temporary queen and propagate constraints
        r, c = next_cell
        tried_temp_queens[board_state] = tried_temp_queens.get(board_state, set())
        tried_temp_queens[board_state].add((r, c))
        if first_temp_queen is None:
            first_temp_queen = (r, c)
        temp_grid[r][c] = 1  # Temporarily place a queen at the most constrained & untried cell
        temp_zeros = propagate_constraints(temp_grid, r, c, variables, grid_colors, permanent_ones, permanent_zeros, temporary=True) # apply rules (set zeros for row, column & around queen)
        backtracking_stack.append((next_cell, set(tried_temp_queens[board_state]), temp_zeros)) # Save state to backtracking stack as temporary

def propagate_constraints(temp_grid, r, c, variables, grid_colors, permanent_ones, permanent_zeros, temporary=False):
    """
    Propagates constraints (place zeros) based on a newly placed queen to reduce search space.

    Returns:
    - A set of positions that were marked as temporary zeros during propagation.
    """

    rows, cols = len(temp_grid), len(temp_grid[0])
    temp_zeros = set() # Track all temporary zeros introduced during this propagation

    # Mark all cells in the same row, column as 0
    for i in range(rows):
        if temp_grid[i][c] is None:  
            temp_grid[i][c] = 0
            if temporary:
                temp_zeros.add((i, c))
    for j in range(cols):
        if temp_grid[r][j] is None:  
            temp_grid[r][j] = 0
            if temporary:
                temp_zeros.add((r, j))

    # Mark immediate diagonal neighbors as 0
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:  # Ensure within bounds
            if temp_grid[nr][nc] is None:
                temp_grid[nr][nc] = 0
                if temporary:
                    temp_zeros.add((nr, nc))

    # Mark all cells of the same color as zeros
    current_color = grid_colors[r][c]
    for i in range(rows):
        for j in range(cols):
            if grid_colors[i][j] == current_color and temp_grid[i][j] is None:
                temp_grid[i][j] = 0
                if temporary:
                    temp_zeros.add((i, j))

    # Block rows or columns if all remaining cells of a color are within them 
    # Similiar to the rule we applied to initalize the grid, but this time we are not grouping, because the computing time is not worth the additonal accuracy (this function runs multiple times every iteration)
    color_cell_positions = {}
    for i in range(rows):
        for j in range(cols):
            color = grid_colors[i][j]
            if temp_grid[i][j] is None:
                if color not in color_cell_positions:
                    color_cell_positions[color] = []
                color_cell_positions[color].append((i, j))

    for color, positions in color_cell_positions.items():
        if len(positions) > 1:
            rows_involved = {pos[0] for pos in positions}
            cols_involved = {pos[1] for pos in positions}
            if len(rows_involved) == 1:  # All cells in the same row
                row = next(iter(rows_involved))
                for col in range(cols):
                    if (row, col) not in positions and temp_grid[row][col] is None: #cells that are NOT the exclusive color
                        temp_grid[row][col] = 0 
                        temp_zeros.add((row, col))
            if len(cols_involved) == 1:  
                col = next(iter(cols_involved))
                for row in range(rows):
                    if (row, col) not in positions and temp_grid[row][col] is None:
                        temp_grid[row][col] = 0
                        temp_zeros.add((row, col))

    # Color dominates: Check for r/c where 100% blank cells are one color => set other cells of that color in other r/c to 0 (queen must be in this r/c)
    for i in range(rows):
        row_colors = [grid_colors[i][j] for j in range(cols) if temp_grid[i][j] is None]
        if len(set(row_colors)) == 1 and row_colors:  # All blanks in the row are of one color
            exclusive_color = row_colors[0]
            for x in range(rows):
                for y in range(cols):
                    if grid_colors[x][y] == exclusive_color and x != i and temp_grid[x][y] is None:
                        temp_grid[x][y] = 0 # Mark as 0 all other cells of that color (that are not in this row)
                        temp_zeros.add((x, y))
                        
    for j in range(cols): # same for cols
        col_colors = [grid_colors[i][j] for i in range(rows) if temp_grid[i][j] is None]
        if len(set(col_colors)) == 1 and col_colors:  
            exclusive_color = col_colors[0]
            for x in range(rows):
                for y in range(cols):
                    if grid_colors[x][y] == exclusive_color and y != j and temp_grid[x][y] is None:
                        temp_grid[x][y] = 0 
                        temp_zeros.add((x, y))
                        

    return temp_zeros

def place_permanent_queens(temp_grid, grid_colors, variables):
    """
    - Places permanent queens and zeros based on the constraints.
    - Ensures that queens and zeros are treated as permanent.
    - Updates the grid directly with permanent constraints.

    Returns:
    - Updated grid with permanent placements.
    """

    rows, cols = len(temp_grid), len(temp_grid[0])
    permanent_zeros = set()
    permanent_ones = set()
    
    # Ensure one queen per color group
    for color in set(grid_colors[r][c] for r in range(rows) for c in range(cols)):
        color_cells = [
            (r, c) for r in range(rows) for c in range(cols)
            if grid_colors[r][c] == color and temp_grid[r][c] is None
        ]
        if len(color_cells) == 1:  # Only one valid cell for this color
            r, c = color_cells[0]
            temp_grid[r][c] = 1  # Place permanent queen
            permanent_ones.add((r, c))
            propagate_constraints(temp_grid, r, c, variables, grid_colors, permanent_ones, permanent_zeros, temporary=False) # will add zeros for r,c and around the queen 

    # One queen per row
    for r in range(rows):
        row_cells = [(r, c) for c in range(cols) if temp_grid[r][c] is None]
        if len(row_cells) == 1:  
            _, c = row_cells[0]
            temp_grid[r][c] = 1  
            permanent_ones.add((r, c))
            propagate_constraints(temp_grid, r, c, variables, grid_colors, permanent_ones, permanent_zeros, temporary=False)

    # Cols
    for c in range(cols):
        col_cells = [(r, c) for r in range(rows) if temp_grid[r][c] is None]
        if len(col_cells) == 1:  
            r, _ = col_cells[0]
            temp_grid[r][c] = 1  
            permanent_ones.add((r, c))
            propagate_constraints(temp_grid, r, c, variables, grid_colors, permanent_ones, permanent_zeros, temporary=False)

    return temp_grid


def reset_temp_queens_and_zeros(temp_grid, temp_queens, temp_zeros, permanent_ones, permanent_zeros):
    """
    - Reset temporary queens (1s) and their associated temp zeros to None.
    - Permanently placed 1s and 0s remain unchanged.
    
    """
    reset_ones = []  # Track temporary queens that are reset
    reset_zeros = []

    # Reset temporary queens (1s)
    for r, c in temp_queens:
        if (r, c) not in permanent_ones:  # Only reset TEMP queens
            temp_grid[r][c] = None
            reset_ones.append((r, c))

    # Reset temporary zeros
    for r, c in temp_zeros:
        if (r, c) not in permanent_zeros:  
            temp_grid[r][c] = None
            reset_zeros.append((r, c))

def is_valid_state(temp_grid, grid_colors):
    
    """
    - Checks if the current grid state is valid based on game rules (constraints).
    - Returns "True" if the state is valid; "False" otherwise.
    """

    rows, cols = len(temp_grid), len(temp_grid[0])

    # Rule 1: One queen per row
    for r in range(rows):
        if sum(cell == 1 for cell in temp_grid[r]) > 1:
            return False

    # Rule 2: One queen per column
    for c in range(cols):
        column = [temp_grid[r][c] for r in range(rows)]  # Extract the column, temp_grid is a 2D list of rows, so to validate we need to extract column manually
        if sum(cell == 1 for cell in column) > 1:
            return False

    # Rule 3: One queen per color
    color_groups = {}
    for r in range(rows):
        for c in range(cols):
            color = grid_colors[r][c]
            if color not in color_groups:
                color_groups[color] = []
            if temp_grid[r][c] == 1:
                color_groups[color].append((r, c))

    for color, queens in color_groups.items():
        if len(queens) > 1:  # More than one queen for a color
            return False

    # Rule 4: No rows, columns, or color groups with only zeros
    for r in range(rows): #rows
        if None not in temp_grid[r] and all(cell == 0 for cell in temp_grid[r]):
            return False

    for c in range(cols): #column
        column = [temp_grid[r][c] for r in range(rows)]
        if None not in column and all(cell == 0 for cell in column):
            return False

    for color in color_groups: #colors
        color_cells = [
            temp_grid[r][c]
            for r in range(rows)
            for c in range(cols)
            if grid_colors[r][c] == color
        ]
        if None not in color_cells and all(cell == 0 for cell in color_cells):
            return False

    # Rule 5: No adjacent queens
    adjacent_directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]
    for r in range(rows):
        for c in range(cols):
            if temp_grid[r][c] == 1:
                for dr, dc in adjacent_directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and temp_grid[nr][nc] == 1:
                        return False

    return True

def is_goal_state(temp_grid):
    """Check if the current grid is fully filled and satisfies all constraints."""
    return all(cell is not None for row in temp_grid for cell in row)

def print_temp_grid(temp_grid):
    """To print the temp grid in a readable format."""
    for row in temp_grid:
        print(" ".join(str(cell) if cell is not None else "." for cell in row))


def print_colored_grid(temp_grid, grid_colors):
    """Print the final solution with colors and format adjustments."""

    # color mapping
    color_mapping = {
        '0': '\033[95m',        # Purple
        '1': '\033[33m',        # Orange
        '2': '\033[94m',        # Blue
        '3': '\033[92m',        # Green
        '4': '\033[97m',        # White/Grey
        '5': '\033[91m',        # Red
        '6': '\033[38;5;226m',  # Bright Yellow
        '7': '\033[38;5;16m',   # Dark Brown
        '8': '\033[38;5;165m',  # Pink
        '9': '\033[96m',        # Cyan
        '10': '\033[38;5;46m',  # Bright Green
        '11': '\033[38;5;51m',  # Bright Blue
        '12': '\033[38;5;208m', # Dark Orange
        '13': '\033[38;5;27m',  # Deep Blue
        '14': '\033[38;5;129m', # Magenta
        '15': '\033[38;5;202m', # Bright Orange
    }
    reset_color = '\033[0m'

    rows, cols = len(temp_grid), len(temp_grid[0])

    for r in range(rows):
        row_display = []
        for c in range(cols):
            value = temp_grid[r][c]
            color = grid_colors[r][c]

            # Replace 1s with X and 0 with - so that queens are easier to identify
            if value == 1:
                cell = f"{color_mapping[color]}X{reset_color}"
            elif value == 0:
                cell = f"{color_mapping[color]}-{reset_color}"
            else:
                cell = f"{color_mapping[color]}.{reset_color}"

            row_display.append(cell)

        print(" ".join(row_display))


def find_most_constrained_cells(temp_grid, grid_colors):
    """
    At the core: Find all viable cells for placing a queen and sort them by the "most constrained" heuristic
    A viable cell is one where the row, column, and color group do not already have a queen
    Most constrained we define as the cell with the fewest available blank cells.
    Therefore we sort all viable cells:
    - Primary: By the minimum number of blank cells in row, column, or color for each cell
    - Secondary (to break ties): Sum of blank cells in row, column, and color
    -> This will let us place the most critical cells first. Even if the placement is wrong, excluding this option (after trying) simplifies the grid strongly
  
    Returns:
    - List of viable cells sorted by how constrained they are
    """
    rows, cols = len(temp_grid), len(temp_grid[0])

    # Helper function to calculate number of constraints for a cell
    def calculate_constraints(r, c):
        # Count blank cells in the row
        row_blank_count = sum(1 for cell in temp_grid[r] if cell is None)
        
        # Cols
        col_blank_count = sum(1 for row in temp_grid if row[c] is None)
        
        # Color
        cell_color = grid_colors[r][c]
        color_blank_count = sum(
            1 for x in range(rows) for y in range(cols)
            if grid_colors[x][y] == cell_color and temp_grid[x][y] is None
        )

        # Calculate primary (minimum) and secondary (sum) sorting metrics
        min_constraint = min(row_blank_count, col_blank_count, color_blank_count) 
        sum_constraint = row_blank_count + col_blank_count + color_blank_count

        return min_constraint, sum_constraint

    # Identify all viable cells
    viable_cells = [
        (r, c) for r in range(rows) for c in range(cols)
        if temp_grid[r][c] is None
        and not any(temp_grid[r][i] == 1 for i in range(cols))  # Row does not have a queen
        and not any(temp_grid[i][c] == 1 for i in range(rows))  # Column 
        and grid_colors[r][c] not in {
            grid_colors[i][j]
            for i in range(rows) for j in range(cols)
            if temp_grid[i][j] == 1
        }  # Color group does not have a queen
    ]

    # Sort viable cells by (min_constraint, and use sum_constraint to break ties)
    viable_cells.sort(key=lambda cell: calculate_constraints(cell[0], cell[1]))

    return viable_cells

# Only needed for online version
def place_queens_with_selenium(driver, solution_grid):
    """
    - automates the process of placing queens on the game page (web interface) using Selenium
    - converts the solution from r,c format back into the HTML cell indexing
    - automatically clicks the respective cells that contain a queen

    Args:
    - driver: Selenium WebDriver instance.
    - solution_grid: 2D list representing the solved puzzle grid.
    """

    rows = len(solution_grid)
    cols = len(solution_grid[0])

    for r in range(rows):
        for c in range(cols):
            if solution_grid[r][c] == 1:  # We need to place a queen here
                cell_selector = f"div[data-cell-idx='{r * cols + c}']"
                cell_element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, cell_selector))
                )

                try:
                    # Double-click the cell to set a Queen (one click would et a cross)
                    cell_element.click()
                    cell_element.click()
                except Exception as e:
                    print(f"Error placing queen at ({r}, {c}): {e}")

# --- Main Script ---
if __name__ == "__main__":

    # Selenium setup
    driver = webdriver.Chrome()
    driver.get("https://www.linkedin.com/login")

    try:
        accept_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept')]")
        ))
        accept_button.click()
    except:
        pass

    # Log into LinkedIn (increase the waiting time if you rather want to log in manually)
    email_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "username"))
    )
    email_field.send_keys(linkedin_email)

    password_field = driver.find_element(By.ID, "password")
    password_field.send_keys(linkedin_password)

    sign_in_button = driver.find_element(By.XPATH, "//button[@aria-label='Sign in']")
    sign_in_button.click()

    # Navigate to the Queens game page
    WebDriverWait(driver, 20).until(
        EC.url_contains("linkedin.com/feed"))
    driver.get("https://www.linkedin.com/games/queens/")

    # Here you have to click the "Start Game" button whenever you feel ready, this will start the algorithm

    try:
        # Wait for the game to load after manual click
        WebDriverWait(driver, 30).until(
            EC.visibility_of_element_located((By.ID, "queens-grid"))
        )

    except Exception as e:
        exit()

    # Start timer when game has loaded
    start_time = time()

    try:
        grid_element = WebDriverWait(driver, 20).until(
            EC.visibility_of_element_located((By.ID, "queens-grid"))
        )
    except:
        driver.quit()
        exit()

    # Get HTML data
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    # Extract grid dimensions and cell details
    grid = soup.find('div', id='queens-grid')
    rows = int(grid['style'].split('--rows: ')[1].split(';')[0])
    cols = int(grid['style'].split('--cols: ')[1].split(';')[0])
    cells = grid.find_all('div', class_='queens-cell-with-border')

    # build grid
    temp_grid = [[None for _ in range(cols)] for _ in range(rows)]
    grid_colors = [[None for _ in range(cols)] for _ in range(rows)]

    for cell in cells:
        data_idx = int(cell['data-cell-idx'])
        cell_color = cell.get('class', [])
        color = None
        for class_name in cell_color:
            if 'cell-color-' in class_name:
                color = class_name.split('-')[-1]
                break

        row = data_idx // cols
        col = data_idx % cols
        grid_colors[row][col] = color

    # Initializes variables
    variables = [[f"x{r+1}{c+1}" for c in range(cols)] for r in range(rows)]

    # Solve the puzzle
    solution = solve_with_backtracking_and_propagation(temp_grid, grid_colors, variables)
   
    if solution:
        end_time = time() # Stop the timer
        place_queens_with_selenium(driver, solution) # Place the queens in web interface
        print(f"Solution found after: {end_time - start_time:.2f} seconds")
        print_colored_grid(solution, grid_colors)

    else:
        print("No solution found.")

    input() # to keep browser open 