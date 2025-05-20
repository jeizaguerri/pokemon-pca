import requests
import pandas as pd

POKEMON_API_ROOT = "https://pokeapi.co/api/v2/pokemon/"
STAT_INDICES = {
            'hp': 0,
            'attack': 1,
            'defense': 2,
            'special-attack': 3,
            'special-defense': 4,
            'speed': 5
        }
OUTPUT_FILE = 'pokemon_data.csv'

def request_api(url):
    try:
        print(f"Fetching data from {url}")
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None

def get_pokemon_stats(pokemon):
    stats = pokemon.get('stats')
    if not stats:
        return None
    
    # Extract relevant stats
    stats_data = [0] * len(STAT_INDICES)
    for stat in stats:
        stat_name = stat.get('stat').get('name')
        base_stat = stat.get('base_stat')
        if stat_name in STAT_INDICES:
            index = STAT_INDICES[stat_name]
            stats_data[index] = base_stat
    return stats_data

def get_pokemon_data(pokemon_url):
    pokemon = request_api(pokemon_url)
    if not pokemon:
        return None
    
    # Extract relevant data
    name = pokemon.get('name')
    id = pokemon.get('id')
    types = [t.get('type').get('name') for t in pokemon.get('types')]
    stats = get_pokemon_stats(pokemon)
    image_url = pokemon.get('sprites').get('front_default')

    # Create a dictionary to hold the Pokemon data
    pokemon_data = {
        'name': name,
        'id': id,
        'type1': types[0],
        'type2': types[1] if len(types) > 1 else None,
        'hp': stats[STAT_INDICES['hp']],
        'attack': stats[STAT_INDICES['attack']],
        'defense': stats[STAT_INDICES['defense']],
        'special-attack': stats[STAT_INDICES['special-attack']],
        'special-defense': stats[STAT_INDICES['special-defense']],
        'speed': stats[STAT_INDICES['speed']],
        'image': image_url
    }
    return pokemon_data


def get_pokemon_urls():
    pokemon_urls = list()

    page = request_api(POKEMON_API_ROOT)
    while page:
        for pokemon in page.get('results'):
            pokemon_urls.append(pokemon.get('url'))
        page = request_api(page.get('next'))
    
    return pokemon_urls

def main():
    # Fetch all Pokémon URLs
    pokemon_urls = get_pokemon_urls()

    # Fetch data for each Pokémon
    pokemon_data = []
    for url in pokemon_urls:
        stats = get_pokemon_data(url)
        if stats:
            pokemon_data.append(stats)
    
    # Save the data to a csv file
    df = pd.DataFrame(pokemon_data)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Data saved to {OUTPUT_FILE}")

        
    



if __name__ == "__main__":
    main()