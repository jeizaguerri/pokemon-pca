import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import itertools

TYPES = set(['grass', 'fire', 'water', 'bug', 'normal', 'poison', 'electric', 'ground',
             'fairy', 'fighting', 'psychic', 'rock', 'ghost', 'ice', 'dragon', 'dark', 'steel',
             'flying'])
STAT_ROWS = ['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']


def count_types(pokemon_data):
    # Count the number of Pokémon with 1 type and 2 types
    one_type_count = pokemon_data[pokemon_data['type2'].isnull()].shape[0]
    two_type_count = pokemon_data[pokemon_data['type2'].notnull()].shape[0]
    return one_type_count, two_type_count


def count_pokemon_by_type(pokemon_data):
    # Count the number of Pokémon for each type
    type_counts = pokemon_data[['type1', 'type2']].melt(value_name='type').dropna()['type'].value_counts()
    return type_counts


def get_type_cooccurrence(pokemon_data):
    # Create a DataFrame to store the co-occurrence counts
    type_cooccurrence = pd.DataFrame(0, index=list(TYPES), columns=list(TYPES))

    # Iterate through each Pokémon and update the co-occurrence counts
    for _, row in pokemon_data.iterrows():
        type1 = row['type1']
        type2 = row['type2']
        if pd.notnull(type2):
            type_cooccurrence.at[type1, type2] += 1
            type_cooccurrence.at[type2, type1] += 1

    return type_cooccurrence


def missing_combinations(type_cooccurrence):
    # Find missing combinations
    missing = []
    for i in range(len(type_cooccurrence)):
        for j in range(i + 1, len(type_cooccurrence)):
            if type_cooccurrence.iloc[i, j] == 0:
                missing.append((type_cooccurrence.index[i], type_cooccurrence.columns[j]))
    return missing


def perform_pca(pokemon_data):
    # Get stats
    stats = pokemon_data[STAT_ROWS].to_numpy()

    # Perform PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(stats)

    # Add PCA result to DataFrame
    pokemon_data['pca1'] = pca_result[:, 0]
    pokemon_data['pca2'] = pca_result[:, 1]
    pokemon_data['pca3'] = pca_result[:, 2]

    return pokemon_data, pca, pca_result


def find_k_nearest_pokemon(pokemon_data, target, k=5):
    # Calculate the distance from the target point
    distances = np.linalg.norm(pokemon_data[['pca1', 'pca2', 'pca3']].to_numpy() - target, axis=1)
    # Get the indices of the k nearest Pokémon
    nearest_indices = np.argsort(distances)[:k]

    # Add the distances to the DataFrame for better visualization
    pokemon_data['distance'] = distances

    # Return the nearest Pokémon
    return pokemon_data.iloc[nearest_indices]


def merge_dataframes(pokemon_data, smogon_data):
    smogon_data['name'] = smogon_data['Pokemon'].str.lower()
    smogon_data['name'] = smogon_data['name'].str.replace(" ", "-", regex=False)
    smogon_data = smogon_data.rename(columns={'Rank': 'rank', 'Usage %': 'usage_percent'})
    smogon_data = smogon_data[['name', 'rank', 'usage_percent']]
    pokemon_data = pd.merge(pokemon_data, smogon_data, on='name', how='left')
    # Set default values for missing rank and usage_percent
    pokemon_data['rank'] = pokemon_data['rank'].fillna(999).astype(int)
    pokemon_data['usage_percent'] = pokemon_data['usage_percent'].fillna(0)
    return pokemon_data


def test_bics(components, data):
    bic_values = []
    for n_components in components:
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(data[['pca1', 'pca2', 'pca3']])
        bic_values.append(gmm.bic(data[['pca1', 'pca2', 'pca3']]))
    
    return bic_values


def train_gmm(data, n_components):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(data[['pca1', 'pca2', 'pca3']])
    return gmm


def generate_pokemon(gmm, samples, pokemon_data):
    generated_points = gmm.sample(samples)[0]

    # Find the closest Pokémon to each generated point
    generated_pokemon = []
    for point in generated_points:
        closest = find_k_nearest_pokemon(pokemon_data, point, k=1)
        generated_pokemon.append(closest.iloc[0])

    generated_pokemon = pd.DataFrame(generated_pokemon).reset_index(drop=True)

    return generated_pokemon


def evaluate_team(team, pokemon_data):
    # Raw stats: sum of all stats
    raw_stats = pokemon_data.loc[team, STAT_ROWS].sum().sum()
    # Coverage: number of unique types (type1 and type2)
    types = set()
    for idx in team:
        row = pokemon_data.loc[idx]
        types.add(row['type1'])
        if pd.notnull(row['type2']):
            types.add(row['type2'])
    coverage = len(types)
    # Balance: average pairwise distance in PCA space
    pca_points = pokemon_data.loc[team, ['pca1', 'pca2', 'pca3']].to_numpy()
    if len(pca_points) > 1:
        dists = [np.linalg.norm(p1 - p2) for p1, p2 in itertools.combinations(pca_points, 2)]
        balance = np.mean(dists)
    else:
        balance = 0
    return raw_stats, coverage, balance


def generate_candidate_teams(pokemon_data, n_teams=1000, team_size=6, usage_threshold=1.0):
    # Only use Pokémon above usage threshold and drop duplicates by name (avoid megas, forms)
    candidates = pokemon_data[pokemon_data['usage_percent'] >= usage_threshold].drop_duplicates('name')
    candidate_indices = candidates.index.tolist()
    teams = []
    for _ in range(n_teams):
        team = np.random.choice(candidate_indices, size=team_size, replace=False)
        teams.append(team)
    return teams


def normalize_scores(scores):
    # Normalize each objective to [0, 1]
    arr = np.array(scores)
    min_vals = arr.min(axis=0)
    max_vals = arr.max(axis=0)
    norm = (arr - min_vals) / (max_vals - min_vals + 1e-8)
    return norm


def select_pareto_front(scores):
    # Find non-dominated solutions (Pareto front)
    arr = np.array(scores)
    is_efficient = np.ones(arr.shape[0], dtype=bool)
    for i, c in enumerate(arr):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(arr[is_efficient] > c, axis=1) | np.all(arr[is_efficient] == c, axis=1)
            is_efficient[i] = True  # Keep self
    return np.where(is_efficient)[0]