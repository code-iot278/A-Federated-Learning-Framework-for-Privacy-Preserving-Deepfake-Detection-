import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# === RhinoCat Optimization (RCO) for Feature Selection ===
# Combines Bobcat Optimization Algorithm (BOA) and Rhinopithecus Swarm Optimization (RSO)

# === Fitness Function ===
def fitness_function(x, data, labels):
    selected_features = x > 0.5
    if np.sum(selected_features) == 0:
        return -np.inf
    try:
        clf = LogisticRegression(max_iter=1000)
        scores = cross_val_score(clf, data[:, selected_features], labels, cv=3, scoring='accuracy')
        return scores.mean()
    except:
        return -np.inf

# === Initialization ===
def initialize_population(n_agents, dim, bounds):
    lower, upper = bounds
    return lower + (upper - lower) * np.random.rand(n_agents, dim)

# === RCO Algorithm ===
def RCO_feature_selection(data, labels, n_agents=30, max_iter=50):
    dim = data.shape[1]
    bounds = (0, 1)
    pop = initialize_population(n_agents, dim, bounds)
    fitness = np.array([fitness_function(ind, data, labels) for ind in pop])
    best_idx = np.argmax(fitness)
    gbest = pop[best_idx].copy()
    gbest_fitness = fitness[best_idx]

    for t in range(max_iter):
        sorted_idx = np.argsort(fitness)[::-1]
        chief = pop[sorted_idx[0]]
        grown = pop[sorted_idx[1:4]]
        juvenile = pop[sorted_idx[4:6]]
        toddler = pop[sorted_idx[6:8]]

        for i in range(n_agents):
            if i < 3:  # Grown SA - Vertical Migration
                alpha = (chief + grown[i]) / 2
                beta = np.abs(chief - grown[i])
                pop[i] = np.random.normal(alpha, beta)
            elif i < 5:  # Juvenile SA - Concerted Search
                tau = (chief + juvenile[i - 3]) / 2
                omega = (grown[0] + juvenile[i - 3]) / 2
                theta = np.abs(chief - juvenile[i - 3])
                xi = np.abs(grown[0] - juvenile[i - 3])
                solujr = (np.random.normal(tau, theta) + np.random.normal(omega, xi)) / 2
                pop[i] = solujr
            elif i < 7:  # Toddler SA - Mimicry
                rho = (grown[0] + toddler[i - 5]) / 2
                lam = (juvenile[0] + toddler[i - 5]) / 2
                phi = np.abs(grown[0] - toddler[i - 5])
                chi = np.abs(juvenile[0] - toddler[i - 5])
                solutd = (np.random.normal(rho, phi) + np.random.normal(lam, chi)) / 2
                pop[i] = solutd
            else:  # Chasing to Catch Prey (BOA Exploitation)
                R = np.random.rand(dim)
                tc = t + 1
                pop[i] += (1 - 2 * R) / (1 + tc) * pop[i]

            # Clip values to bounds
            pop[i] = np.clip(pop[i], *bounds)

        # Fitness Update
        fitness = np.array([fitness_function(ind, data, labels) for ind in pop])
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > gbest_fitness:
            gbest = pop[best_idx].copy()
            gbest_fitness = fitness[best_idx]

        print(f"Iteration {t+1}/{max_iter} | Best Accuracy: {gbest_fitness:.5f}")

    selected_mask = gbest > 0.5
    return selected_mask, gbest_fitness

# === Main Execution ===
if __name__ == "__main__":
    input_csv = ""
    output_csv = ""
    label_column = 'label'  # replace with actual label column name

    df = pd.read_csv(input_csv)
    X = df.drop(columns=[label_column]).values
    y = df[label_column].values

    selected_mask, best_score = RCO_feature_selection(X, y, n_agents=30, max_iter=50)

    selected_columns = df.drop(columns=[label_column]).columns[selected_mask]
    selected_df = df[selected_columns]
    selected_df[label_column] = y
    selected_df.to_csv(output_csv, index=False)

    print(f"\n✅ Selected features saved to: {output_csv}")
    print(f"✔️ Selected Features: {list(selected_columns)}")