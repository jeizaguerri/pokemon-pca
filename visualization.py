import seaborn as sns
import plotly.express as px
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app_logic import find_k_nearest_pokemon


TYPE_COLORS = {
    'normal': '#A8A77A', 'fire': '#EE8130', 'water': '#6390F0', 'electric': '#F7D02C',
    'grass': '#7AC74C', 'ice': '#96D9D6', 'fighting': '#C22E28', 'poison': '#A33EA1',
    'ground': '#E2BF65', 'flying': '#A98FF3', 'psychic': '#F95587', 'bug': '#A6B91A',
    'rock': '#B6A136', 'ghost': '#735797', 'dragon': '#6F35FC', 'dark': '#705746',
    'steel': '#B7B7CE', 'fairy': '#D685AD'
}
STAT_ROWS = ['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']


def plot_count_types(one_type_count, two_type_count):
    # Plot the counts of Pokémon with 1 type and 2 types using seaborn
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        x=['1 Type', '2 Types'],
        y=[one_type_count, two_type_count],
        palette=['#FF9999', '#66B3FF'],
        ax=ax,
        hue=['1 Type', '2 Types'],
    )
    ax.set_title('Count of Pokémon by Type Count', fontsize=16, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xlabel('Type Count', fontsize=12)
    sns.despine()
    return fig


def plot_pokemon_by_type(type_counts):
    # Plot the counts of Pokémon by type using seaborn
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=type_counts.index,
        y=type_counts.values,
        palette=[TYPE_COLORS.get(t, '#000000') for t in type_counts.index],
        ax=ax,
        hue=type_counts.index,
    )
    ax.set_title('Count of Pokémon by Type', fontsize=16, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xlabel('Type', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    sns.despine()
    return fig


def plot_type_cooccurrence(type_cooccurrence):
    # Plot the co-occurrence matrix
    mask = np.triu(np.ones_like(type_cooccurrence))
    
    fig = plt.figure(figsize=(10, 8))
    g = sns.heatmap(type_cooccurrence, annot=True, fmt='d', cmap='Blues', cbar=False, mask=mask)
    g.set_title('Type Co-occurrence Matrix')
    g.set_xlabel('Type 2')
    g.set_ylabel('Type 1')

    for val,label in enumerate(g.get_xticklabels()):
        color = TYPE_COLORS.get(label.get_text(), '#000000')
        label.set_color(color)
    
    for val,label in enumerate(g.get_yticklabels()):
        color = TYPE_COLORS.get(label.get_text(), '#000000')
        label.set_color(color)

    return fig


def plot_pca_component_loadings(pca):
    loadings = pd.DataFrame(
        pca.components_[:3],  # First 3 PCs
        columns=STAT_ROWS,
        index=[f'PC{i+1}' for i in range(3)]
    )
    
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(loadings.T, annot=True, cmap='coolwarm', center=0)
    plt.title("PCA Component Loadings (Contributions of Stats to PCs)")
    plt.ylabel("Stats")
    plt.xlabel("Principal Components")
    plt.tight_layout()
    
    return fig


def plot_pca(pokemon_data):
    fig = px.scatter_3d(
        pokemon_data,
        x='pca1',
        y='pca2',
        z='pca3',
        color='type1',
        color_discrete_map=TYPE_COLORS,
        title='3D PCA of Pokémon Stats',
        labels={'pca1': 'PCA Component 1', 'pca2': 'PCA Component 2', 'pca3': 'PCA Component 3'},
        hover_data=['name'] if 'name' in pokemon_data.columns else None,
        width=800,  # Set the width of the plot
        height=600,  # Set the height of the plot
    )
    fig.update_traces(marker_size = 5)

    return fig


def plot_convex_hull(pca_result, pokemon_data):
    fig = go.Figure()

    # Add convex hulls per type
    for t, group in pokemon_data.groupby('type1'):
        if len(group) < 5:
            continue  # not enough for a 3D hull
        points = group[['pca1', 'pca2', 'pca3']].values
        centroid = points.mean(axis=0)
        dists = np.linalg.norm(points - centroid, axis=1)
        cutoff = np.percentile(dists, 80)
        trimmed_points = points[dists <= cutoff]

        try:
            hull = ConvexHull(trimmed_points)
            simplices = hull.simplices

            # Get vertices of triangles in the hull
            x_hull = []
            y_hull = []
            z_hull = []
            i_list, j_list, k_list = [], [], []

            for s in simplices:
                i, j, k = s
                i_list.append(i)
                j_list.append(j)
                k_list.append(k)

            fig.add_trace(go.Mesh3d(
                x=trimmed_points[:, 0],
                y=trimmed_points[:, 1],
                z=trimmed_points[:, 2],
                i=i_list,
                j=j_list,
                k=k_list,
                name=t,
                opacity=0.15,
                color=TYPE_COLORS.get(t, '#999999'),
                hoverinfo='name',  # Show type name on hover
                showscale=False,
                legendgroup=t,
                showlegend=True  # Show this in the legend
            ))

        except Exception as e:
            st.warning(f"Could not compute hull for {t}: {e}")

    # Layout
    fig.update_layout(
        title="Convex Hulls of Pokémon Types in PCA Space",
        scene=dict(
            xaxis_title="PCA1",
            yaxis_title="PCA2",
            zaxis_title="PCA3",
        ),
        width=1000,
        height=800,
        showlegend=True
    )

    return fig


def plot_pca_usage(filtered_pokemon_data):
    fig = px.scatter_3d(
        filtered_pokemon_data,
        x='pca1',
        y='pca2',
        z='pca3',
        color='usage_percent',
        color_continuous_scale=px.colors.sequential.Viridis,
        title='3D PCA of Pokémon Stats (Filtered by Usage)',
        labels={'pca1': 'PCA Component 1', 'pca2': 'PCA Component 2', 'pca3': 'PCA Component 3'},
        hover_data=['name'] if 'name' in filtered_pokemon_data.columns else None,
        width=800,  # Set the width of the plot
        height=600,  # Set the height of the plot
    )

    # Make Pokémon with usage_percent > 1% bigger
    marker_sizes = np.where(filtered_pokemon_data['usage_percent'] > 1, 15, 5)
    fig.update_traces(marker_size=marker_sizes)

    return fig


def find_and_plot_k_nearest(pokemon_data, target, k=5):
    closest = find_k_nearest_pokemon(pokemon_data, target, k)
    closest = closest[['image', 'name', 'distance']]

    st.dataframe(
        closest,
        column_config={
            'image': st.column_config.ImageColumn(label="Image", width=20),
            'name': st.column_config.TextColumn(label="Name"),
            'distance': st.column_config.ProgressColumn(
                label="Distance",
                max_value=float(closest['distance'].max()),
                format="%.2f"
            ),
        },
        hide_index=True,
        use_container_width=True
    )

def plot_bic(components, bic_values):
    fig = px.line(
        x=components,
        y=bic_values,
        labels={'x': 'Number of Components', 'y': 'BIC'},
        title='BIC vs Number of Components',
    )
    fig.update_traces(mode='lines+markers')
    return fig


def display_generated_pokemon(generated_pokemon, n_columns):
    # Display generated Pokémon in a 5x2 grid (images and names)
    cols = st.columns(n_columns)
    for i, row in generated_pokemon.iterrows():
        with cols[i % n_columns]:
            st.image(row['image'], use_container_width=True)
            st.caption(row['name'].capitalize())


def show_teams(pareto_teams, pareto_scores, pokemon_data):
    for i, (team, (raw_stats, coverage, balance)) in enumerate(zip(pareto_teams, pareto_scores)):
        st.markdown(f"#### Team {i+1}")
        st.markdown(f"**Raw stats:** {int(raw_stats)} &nbsp; | &nbsp; **Coverage:** {coverage} types &nbsp; | &nbsp; **Balance:** {balance:.2f}")
        team_df = pokemon_data.loc[team][['image', 'name', 'type1', 'type2'] + STAT_ROWS]
        st.dataframe(
            team_df,
            column_config={
                'image': st.column_config.ImageColumn(label="Image", width=20),
                'name': st.column_config.TextColumn(label="Name"),
                'type1': st.column_config.TextColumn(label="Type 1"),
                'type2': st.column_config.TextColumn(label="Type 2"),
            },
            hide_index=True,
            use_container_width=True
        )
