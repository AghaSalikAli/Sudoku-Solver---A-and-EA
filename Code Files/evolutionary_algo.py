import numpy as np
import random
import time


# Evolutionary Algorithm Functions
def mutate(individual, sudoku, mutation_rate):
    new_individual = individual.copy()
    for i in range(9):
        for j in range(9):
            if random.random() < mutation_rate and sudoku[i][j] == 0:
                new_individual[i][j] = random.randint(1, 9)
    return new_individual


def valid_move(board, row, col, number):
    if number in board[row] or number in board[:, col]:
        return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i][j] == number:
                return False
    return True


def create_individual(sudoku):
    individual = np.array(sudoku)
    for i in range(9):
        missing = set(range(1, 10)) - set(individual[i])
        np.place(individual[i], individual[i] == 0, list(missing))
    return individual


def compute_fitness(individual):
    fitness = 0
    # Penalize row conflicts
    for i in range(9):
        fitness += len(set(individual[i]))
    # Penalize column conflicts
    for j in range(9):
        fitness += len(set(individual[:, j]))
    # Penalize block conflicts
    for i in range(3):
        for j in range(3):
            block = individual[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3]
            fitness += len(set(block.flatten()))
    return fitness


def crossover(parent1, parent2):
    point = random.randint(1, 8)
    child1 = np.vstack((parent1[:point, :], parent2[point:, :]))
    child2 = np.vstack((parent2[:point, :], parent1[point:, :]))
    return child1, child2



def select_parents(population, fitnesses):
    selected = random.choices(population, weights=fitnesses, k=2)
    return selected[0], selected[1]

def evolution_strategy(sudoku, population_size, iterations, mutation_rate, speed, st, update_ui=True):
    population = [create_individual(sudoku) for _ in range(population_size)]
    best_individual = max(population, key=compute_fitness)
    best_fitness = compute_fitness(best_individual)

    if update_ui:
        table_placeholder = st.empty()
        generation_text = st.empty()
        accuracy_text = st.empty()
        final_metrics_text = st.empty()
        progress_bar = st.progress(0)

    for gen in range(1, iterations + 1):
        new_population = []
        fitnesses = [compute_fitness(ind) for ind in population]

        # Elitism: Carry over the best individual
        new_population.append(best_individual)

        for _ in range((population_size - 1) // 2):
            parent1, parent2 = select_parents(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, sudoku, mutation_rate))
            new_population.append(mutate(child2, sudoku, mutation_rate))

        population = sorted(new_population + population, key=compute_fitness, reverse=True)[:population_size]
        current_best = population[0]
        current_fitness = compute_fitness(current_best)

        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_individual = current_best

        accuracy = best_fitness / (9 * 9 * 3)

        if update_ui:
            # Update Streamlit UI
            table_placeholder.table(best_individual)
            generation_text.text(f"Generation: {gen}/{iterations}")
            accuracy_text.text(f"Accuracy: {accuracy:.2%}")
            progress_bar.progress(gen / iterations)

        if accuracy == 1.0:
            if update_ui:
                final_metrics_text.text(
                    f"Final Accuracy: {accuracy:.2%}\nGeneration: {gen}\nMutation Rate: {mutation_rate}\nPopulation Size: {population_size}")
                st.success("Sudoku Solved!")
            return accuracy, gen

        time.sleep(speed)

    if update_ui:
        final_metrics_text.text(
            f"Final Accuracy: {accuracy:.2%}\nGeneration: {iterations}\nMutation Rate: {mutation_rate}\nPopulation Size: {population_size}")

    if update_ui:
        st.warning("Couldn't find the solution with the current parameters", icon="❌")
    return accuracy, iterations



