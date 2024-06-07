import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Importing algorithms
from evolutionary_algo import evolution_strategy
from astar_algo import a_star_algorithm, a_star_analysis


def parse_sudoku(file_content):
    lines = file_content.strip().split('\n')
    sudoku = []
    for line in lines:
        row = []
        for char in line.strip():
            if char == '-':
                row.append(0)
            elif char.isdigit():
                row.append(int(char))
        sudoku.append(row)
    return sudoku


def run_evolutionary_analysis(sudoku):
    configs = [
        {"population_size": 50, "iterations": 100, "mutation_rate": 0.05},
        {"population_size": 50, "iterations": 100, "mutation_rate": 0.1},
        {"population_size": 50, "iterations": 100, "mutation_rate": 0.15},
        {"population_size": 50, "iterations": 100, "mutation_rate": 0.20},
        {"population_size": 100, "iterations": 100, "mutation_rate": 0.05},
        {"population_size": 100, "iterations": 100, "mutation_rate": 0.1},
        {"population_size": 100, "iterations": 100, "mutation_rate": 0.15},
        {"population_size": 100, "iterations": 100, "mutation_rate": 0.2},
        {"population_size": 200, "iterations": 100, "mutation_rate": 0.05},
        {"population_size": 200, "iterations": 100, "mutation_rate": 0.1},
        {"population_size": 200, "iterations": 100, "mutation_rate": 0.15},
        {"population_size": 200, "iterations": 100, "mutation_rate": 0.2},
    ]

    results = []

    total_tasks = len(configs)  # Total number of tasks to be completed
    task_counter = 0  # Counter to keep track of completed tasks

    progress_bar = st.progress(0)  # Initialize the progress bar

    with st.spinner("Running analysis..."):
        for config in configs:
            population_size = config["population_size"]
            iterations = config["iterations"]
            mutation_rate = config["mutation_rate"]
            accuracy, generation = evolution_strategy(sudoku, population_size, iterations, mutation_rate, 0, st,
                                                      update_ui=False)
            results.append({
                "Population Size": population_size,
                "Generations": generation,
                "Mutation Rate": mutation_rate,
                "Accuracy": accuracy,
            })

            task_counter += 1
            progress_bar.progress(task_counter / total_tasks)  # Update progress bar

    df = pd.DataFrame(results)
    st.table(df)

    # Plotting the results
    fig, ax = plt.subplots()
    for key, grp in df.groupby(['Population Size']):
        ax = grp.plot(ax=ax, kind='line', x='Mutation Rate', y='Accuracy', label=f"Population {key}")
    plt.title('Accuracy vs Mutation Rate for Different Population Sizes')
    plt.xlabel('Mutation Rate')
    plt.ylabel('Accuracy')
    st.pyplot(fig)

def run_a_star(sudoku, heuristic_choice):
    table_placeholder = st.empty()

    def update_ui(current_state):
        table_placeholder.table(current_state)

    result, time_taken, configurations_explored = a_star_algorithm(sudoku, heuristic_choice, update_ui)
    if result:
        st.success(f"Sudoku solved in {time_taken:.2f} seconds and {configurations_explored} configurations explored")
        table_placeholder.table(result)
    else:
        st.error("No solution found.")


def run_astar_analysis(prefed_flag, sudoku):
    if prefed_flag == True:
        sudokus = [
            # Sudoku 1
            [
                [0, 0, 0, 2, 6, 0, 7, 0, 1],
                [6, 8, 0, 0, 7, 0, 0, 9, 0],
                [1, 9, 0, 0, 0, 4, 5, 0, 0],
                [8, 2, 0, 1, 0, 0, 0, 4, 0],
                [0, 0, 4, 6, 0, 2, 9, 0, 0],
                [0, 5, 0, 0, 0, 3, 0, 2, 8],
                [0, 0, 9, 3, 0, 0, 0, 7, 4],
                [0, 4, 0, 0, 5, 0, 0, 3, 6],
                [7, 0, 3, 0, 1, 8, 0, 0, 0]
            ]
            ,

            [
                [5, 3, 0, 0, 7, 0, 0, 0, 0],
                [6, 0, 0, 1, 9, 5, 0, 0, 0],
                [0, 9, 8, 0, 0, 0, 0, 6, 0],
                [8, 0, 0, 0, 6, 0, 0, 0, 3],
                [4, 0, 0, 8, 0, 3, 0, 0, 1],
                [7, 0, 0, 0, 2, 0, 0, 0, 6],
                [0, 6, 0, 0, 0, 0, 2, 8, 0],
                [0, 0, 0, 4, 1, 9, 0, 0, 5],
                [0, 0, 0, 0, 8, 0, 0, 7, 9]
            ]
            ,

            [
                [1, 0, 0, 0, 0, 7, 0, 9, 0],
                [0, 3, 0, 0, 2, 0, 0, 0, 8],
                [0, 0, 9, 6, 0, 0, 5, 0, 0],
                [0, 0, 5, 3, 0, 0, 9, 0, 0],
                [0, 1, 0, 0, 8, 0, 0, 0, 2],
                [6, 0, 0, 0, 0, 4, 0, 0, 0],
                [3, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 4, 0, 0, 0, 0, 0, 0, 7],
                [0, 0, 7, 0, 0, 0, 3, 0, 0]
            ]
            ,

            [
            [0, 0, 0, 0, 0, 0, 0, 8, 0],
            [0, 8, 0, 0, 4, 6, 0, 0, 2],
            [0, 0, 3, 0, 2, 8, 0, 0, 0],
            [4, 0, 5, 0, 0, 7, 0, 2, 6],
            [2, 0, 0, 0, 5, 0, 0, 0, 4],
            [6, 3, 0, 2, 0, 0, 1, 0, 7],
            [0, 0, 0, 8, 6, 0, 2, 0, 0],
            [8, 0, 0, 3, 1, 0, 0, 4, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0]],


            [
                [5, 3, 0, 0, 7, 0, 0, 0, 0],
                [6, 0, 0, 1, 9, 5, 0, 0, 0],
                [0, 9, 8, 0, 0, 0, 0, 6, 0],
                [8, 0, 0, 0, 6, 0, 0, 0, 3],
                [4, 0, 0, 8, 0, 3, 0, 0, 1],
                [7, 0, 0, 0, 2, 0, 0, 0, 6],
                [0, 6, 0, 0, 0, 0, 2, 8, 0],
                [0, 0, 0, 4, 1, 9, 0, 0, 5],
                [0, 0, 0, 0, 8, 0, 0, 7, 9]
            ]
        ]
    else:
        sudokus = [sudoku]

    results = []
    total_tasks = len(sudokus) * 2  # Total number of tasks to be completed
    task_counter = 0  # Counter to keep track of completed tasks

    progress_bar = st.progress(0)  # Initialize the progress bar



    with st.spinner("Running A* analysis..."):
        for i, sudoku in enumerate(sudokus, 1):
            for heuristic in [1, 2]:
                _, time_taken, configurations_explored = a_star_analysis(sudoku, heuristic)
                results.append({
                    "Sudoku": f"Sudoku {i}",
                    "Heuristic": f"Heuristic {heuristic}",
                    "Time Taken (s)": time_taken,
                    "Configurations Explored": configurations_explored
                })
                task_counter += 1
                progress_bar.progress(task_counter / total_tasks)


    df = pd.DataFrame(results)
    st.table(df)

    if prefed_flag == True:
        # Plotting the results
        fig, ax = plt.subplots()
        for heuristic in [1, 2]:
            subset = df[df["Heuristic"] == f"Heuristic {heuristic}"]
            ax.plot(subset["Sudoku"], subset["Configurations Explored"], label=f"Heuristic {heuristic}")
        plt.title('Configurations Explored by A* Heuristic')
        plt.xlabel('Sudoku')
        plt.ylabel('Configurations Explored')
        plt.legend()
        st.pyplot(fig)


def get_custom_sudoku(unique_key):
    sudoku = None
    uploaded_file = st.file_uploader("Upload a Sudoku file", type=["txt"], key=unique_key)
    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8")
        sudoku = parse_sudoku(file_content)
        st.write("Your Uploaded Sudoku:")

    if sudoku:
        st.table(sudoku)
    return sudoku


astarchoice = 0
analysis_flag = False

placeholder_sudoku = [
            [1, 0, 0, 0, 0, 7, 0, 9, 0],
            [0, 3, 0, 0, 2, 0, 0, 0, 8],
            [0, 0, 9, 6, 0, 0, 5, 0, 0],
            [0, 0, 5, 3, 0, 0, 9, 0, 0],
            [0, 1, 0, 0, 8, 0, 0, 0, 2],
            [6, 0, 0, 0, 0, 4, 0, 0, 0],
            [3, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 4, 0, 0, 0, 0, 0, 0, 7],
            [0, 0, 7, 0, 0, 0, 3, 0, 0]
        ]

st.title("Sudoku Solver")

option = st.radio(
    "Choose an option:",
    ("Do nothing",'Solve with a specific algorithm', 'Perform Analysis')
)

if option == 'Solve with a specific algorithm':
    sudoku_type = st.selectbox(
        "Choose Sudoku type:",
        ("None", "Use placeholder Sudoku", "Upload a Sudoku file")
    )
    if sudoku_type == "None":
        st.warning("Invalid choice. Please choose a Sudoku type.")
        sudoku_flag = False
    elif sudoku_type == "Use placeholder Sudoku":
        sudoku = placeholder_sudoku
        sudoku_flag = True
        st.write("Placeholder Sudoku:")
        st.table(sudoku)
    else:
        sudoku = get_custom_sudoku("solve_with_specific_algo")
        sudoku_flag = True

    if sudoku_flag:
        algorithm = st.selectbox(
            "Choose Algorithm",
            ["Evolutionary Algorithm", "A* Algorithm"]
        )
        if algorithm == "A* Algorithm":
            astarchoice_text = st.selectbox(
                "Choose an option:",
                ('Heuristic 1 (Number of empty cells)', 'Heuristic 2 (Number of conflicts)')
            )

            if astarchoice_text == "Heuristic 1 (Number of empty cells)":
                astarchoice = 1
            else:
                astarchoice = 2
            if st.button('Start Solving with A*'):
                run_a_star(sudoku, astarchoice)
        elif algorithm == "Evolutionary Algorithm":
            population_size = st.sidebar.slider("Population Size", min_value=10, max_value=100, value=50, step=10)
            iterations = st.sidebar.slider("Iterations", min_value=10, max_value=500, value=100, step=10)
            mutation_rate = st.sidebar.slider("Mutation Rate", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
            speed = st.sidebar.slider("Select Speed (seconds)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
            if st.button('Start Solving with Evolutionary Algorithm'):
                evolution_strategy(sudoku, population_size, iterations, mutation_rate, speed, st, update_ui=True)

elif option == 'Perform Analysis':
    sudoku_type = st.selectbox(
        "Choose Sudoku Type for Analysis:",
        ("None", "Use pre-fed Sudoku(s)", "Upload a Sudoku file")
    )
    if sudoku_type == "None":
        analysis_flag = False
        st.warning("Invalid choice. Please choose a Sudoku type for analysis.")
    elif sudoku_type == "Use pre-fed Sudoku(s)":
        analysis_flag = True
        st.write("Using pre-fed Sudoku(s):")
    else:
        analysis_flag = True

    if analysis_flag:
        if sudoku_type=="Use pre-fed Sudoku(s)":
            sudoku = placeholder_sudoku
            st.write("Choosing A* Algorithm will run five pre-fed sudokus comparing heuristics. Choosing Evolutionary Algorithm will run a single pre-fed sudoku with different parameters.")
            analysis_algo = st.selectbox(
                "Choose Algorithm for Analysis",
                ["Evolutionary Algorithm", "A* Algorithm"]
            )
            if analysis_algo == "Evolutionary Algorithm":
                if st.button('Start Evolutionary Analysis'):
                    st.subheader("Running Evolutionary Algorithm Analysis on Pre-fed Sudoku...")
                    run_evolutionary_analysis(sudoku)
            elif analysis_algo == "A* Algorithm":
                if st.button('Start A* Analysis'):
                    st.subheader("Running A* Algorithm Analysis on five Pre-fed Sudokus...")
                    run_astar_analysis(True, None)

        elif sudoku_type == "Upload a Sudoku file":
            sudoku = get_custom_sudoku("perform_analysis")
            st.write("Choosing A* Algorithm will run the uploaded sudoku comparing heuristics. Choosing Evolutionary Algorithm will run the uploaded sudoku with different parameters.")
            analysis_algo = st.selectbox(
                "Choose Algorithm for Analysis",
                ["Evolutionary Algorithm", "A* Algorithm"]
            )
            if analysis_algo == "Evolutionary Algorithm":
                if st.button('Start Evolutionary Analysis'):
                    st.subheader("Running Evolutionary Algorithm Analysis on Uploaded Sudoku...")
                    run_evolutionary_analysis(sudoku)
            elif analysis_algo == "A* Algorithm":
                if st.button('Start A* Analysis'):
                    st.subheader("Running A* Algorithm Analysis on Uploaded Sudoku...")
                    run_astar_analysis(False, sudoku)

else: 
        st.subheader("Who cares about sudokus? :)")
