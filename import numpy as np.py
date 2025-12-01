import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

def abc_search(X, y, pop_size=6, iter_max=10, cv=3, random_state=42):
    rng = np.random.RandomState(random_state)

    hidden_choices = [(64, 32), (128, 64), (32, 16)]
    alpha_choices = [1e-4, 1e-3, 1e-2]

    population = []
    for _ in range(pop_size):
        h = hidden_choices[rng.randint(len(hidden_choices))]
        a = alpha_choices[rng.randint(len(alpha_choices))]
        population.append((h, a))

    def fitness(params):
        h, a = params
        model = MLPClassifier(hidden_layer_sizes=h, alpha=a, max_iter=200, random_state=0)
        try:
            return cross_val_score(model, X, y, cv=cv, scoring="f1").mean()
        except:
            return 0.0

    best, best_score = None, -1
    for it in range(iter_max):
        scored = [(p, fitness(p)) for p in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        if scored[0][1] > best_score:
            best, best_score = scored[0]

        new_pop = [scored[0][0]]
        while len(new_pop) < pop_size:
            h = hidden_choices[rng.randint(len(hidden_choices))]
            a = alpha_choices[rng.randint(len(alpha_choices))]
            new_pop.append((h, a))

        population = new_pop

    return {"hidden_layer_sizes": best[0], "alpha": best[1], "score": best_score}
