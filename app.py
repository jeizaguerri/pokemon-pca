import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
from scipy.spatial import ConvexHull
import plotly.graph_objects as go

os.chdir(os.path.dirname(os.path.abspath(__file__)))

INPUT_DATA_FILE = 'pokemon_data.csv'
TYPE_COLORS = {
    'normal': '#A8A77A', 'fire': '#EE8130', 'water': '#6390F0', 'electric': '#F7D02C',
    'grass': '#7AC74C', 'ice': '#96D9D6', 'fighting': '#C22E28', 'poison': '#A33EA1',
    'ground': '#E2BF65', 'flying': '#A98FF3', 'psychic': '#F95587', 'bug': '#A6B91A',
    'rock': '#B6A136', 'ghost': '#735797', 'dragon': '#6F35FC', 'dark': '#705746',
    'steel': '#B7B7CE', 'fairy': '#D685AD'
}
TYPES = set(['grass', 'fire', 'water', 'bug', 'normal', 'poison', 'electric', 'ground',
             'fairy', 'fighting', 'psychic', 'rock', 'ghost', 'ice', 'dragon', 'dark', 'steel',
             'flying'])
STAT_ROWS = ['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']


def count_types(pokemon_data):
    # Count the number of Pokémon with 1 type and 2 types
    one_type_count = pokemon_data[pokemon_data['type2'].isnull()].shape[0]
    two_type_count = pokemon_data[pokemon_data['type2'].notnull()].shape[0]
    return one_type_count, two_type_count

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

def count_pokemon_by_type(pokemon_data):
    # Count the number of Pokémon for each type
    type_counts = pokemon_data[['type1', 'type2']].melt(value_name='type').dropna()['type'].value_counts()
    return type_counts

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

def find_k_nearest_pokemon(pokemon_data, target, k=5):
    # Calculate the distance from the target point
    distances = np.linalg.norm(pokemon_data[['pca1', 'pca2', 'pca3']].to_numpy() - target, axis=1)
    # Get the indices of the k nearest Pokémon
    nearest_indices = np.argsort(distances)[:k]

    # Add the distances to the DataFrame for better visualization
    pokemon_data['distance'] = distances

    # Return the nearest Pokémon
    return pokemon_data.iloc[nearest_indices]

def find_and_plot_k_nearest(pokemon_data, target, k=5):
    closest = find_k_nearest_pokemon(pokemon_data, target, k)
    closest = closest[['image', 'name', 'distance']]

    st.data_editor(
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

st.set_page_config(layout="centered")  # Ensure proper layout

st.title("Pokémon Data Analysis in Low-Dimensional Space")
st.subheader("By Juan Eizaguerri")
st.image("imgs/banner.png")
st.markdown("""
    Visualizing data in a graphical way often helps to understand the data better, however, this is not easy to do when dealing with high dimensional data.
    I wanted to put this into practice and thought that it would be a good idea to use data I am already familiar with, given that I should be able to tell when a result is wrong or unexpected.
    This is how I ended up with a needlessly complex (but fun) analysis on Pokémon stats and types. Are there type combinations yet to be explored? Could they be classified in large groups? Do Pokémon stats correlate with their types? Could we use this information to build a better team? I will try to answer these questions in this project.
    """)

st.info("""
    NOTE: Most of the plots and results here are computed in real time to be fully interactive. This means some of the plots might take some time to load and you could encounter some errors. If you do, please feel free to contact me and I will try to fix it as soon as possible.
    """)

st.header("1. Input data")
st.markdown("""
    Before starting to process and analyze the data, we first need to load it from somewhere. Thankfully, the people from [PokéAPI](https://pokeapi.co/) are doing an exceptional job at providing Pokémon data in a structured way.
    A few simple requets to their API and we can get all the data we need. For this project, I only wanted to get the Pokémon stats and types, although arguably, other data such as abilities, moves, and evolutions could be interesting too to answer some of the questions I proposed in the introduction.
    It wouldn't be too polite to request all the data every time we want to analyze it, so I decided to save the data in a [CSV](https://github.com/jeizaguerri/pokemon-pca/blob/main/pokemon_data.csv) file which you can find in the same directory as this notebook.
            """)

if not os.path.exists(INPUT_DATA_FILE):
    st.error(f"Input data file '{INPUT_DATA_FILE}' not found.")
    st.stop()

st.success("Data file found!")
try:
    pokemon_data = pd.read_csv(INPUT_DATA_FILE)
    st.write("You can see a preview of the data below:")
    st.dataframe(pokemon_data)
except Exception as e:
    st.error(f"Failed to load data: {e}")

st.header("2. Pokemon types distribution")
st.markdown("""
    Let's start by the basics. Types are going to be the main classification we are going to use in this project, so it is important to understand how they are distributed.
            """)

if 'type1' not in pokemon_data.columns or 'type2' not in pokemon_data.columns:
    st.stop("Data does not contain 'type1' or 'type2' columns.")

one_type_count, two_type_count = count_types(pokemon_data)
fig1 = plot_count_types(one_type_count, two_type_count)
st.pyplot(fig1)
plt.close(fig1)
st.markdown("""
    Pokémon can have one or two types. The first type is the main type, while the second type is the secondary type. As you can see in the plot above, most Pokémon have two types, while a still significant number of Pokémon have only one type.
    This is important to keep in mind when analyzing the data, as we are going to be filtering the data by type in future sections. In this cases we will only be using the first type, as the second type is not always available.
    
    Next, let's see how many Pokémon there are of each type (having main and secondary type in mind), which is to understand the distribution of types in the data and to see if there are any types that are over or under represented.
    """)

type_counts = count_pokemon_by_type(pokemon_data)
fig2 = plot_pokemon_by_type(type_counts)
st.pyplot(fig2)
plt.close(fig2)
st.markdown("""
    While there is is a good amount of Pokémon of each type, it is surprising to see such an unbalanced distribution. It makes sense that the later introduced types such as fairy have less Pokémon, but it is interesting to see that some types such as ice, which is a type that was introduced in the first generation, have so few Pokémon, global warming seems to be affecting Pokémon too.

    Let's dig a bit deeper and see if there are more common combinations of types, or if there are types combinations that are yet to be explored. For this, we are going to create a co-occurrence matrix of types. The upper triangle of the matrix will be redundant so we will only show the lower triangle, also, the main diagonal is removed since there are no Pokémon with the same type twice.
            """)

type_cooccurrence = get_type_cooccurrence(pokemon_data)
fig3 = plot_type_cooccurrence(type_cooccurrence)
st.pyplot(fig3)
plt.close(fig3)

st.markdown("""
    Good news for normal-flying fans, you can fill a full box of them in your [PC](https://bulbapedia.bulbagarden.net/wiki/Pok%C3%A9mon_Storage_System#Limitations). Overall, most of the combinations are covered, but there are exactly 9 combinations that are yet to be explored. Here is the full list:
""")
missing = missing_combinations(type_cooccurrence)
# Display missing combinations in a compact way
combos = []
for t1, t2 in missing:
    color1 = TYPE_COLORS.get(t1, "#000000")
    color2 = TYPE_COLORS.get(t2, "#000000")
    combos.append(
        f"<span style='color:{color1};font-weight:bold'>{t1.capitalize()}</span>-"
        f"<span style='color:{color2};font-weight:bold'>{t2.capitalize()}</span>"
    )
st.markdown(", ".join(combos), unsafe_allow_html=True)
        
st.header("3. Performing Analysis on a PCA Space")
st.markdown("""
    The next step is to perform dimensionality reduction on the stats data. For this, we are going to use [Principal Component Analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis). Essentially, PCA is a technique that transforms the data into a lower dimensional space while preserving as much variance as possible. There are [other dimensionality reduction techniques](https://www.researchgate.net/profile/Eric-Postma/publication/228657549_Dimensionality_Reduction_A_Comparative_Review/links/0046353a3047fc2863000000/Dimensionality-Reduction-A-Comparative-Review.pdf) such as t-SNE and UMAP, but I like to use PCA as a starting point since it is unsupervised easy to interpret.
            """)

pokemon_data, pca, pca_result = perform_pca(pokemon_data)
st.markdown("**PCA explained variance ratio:**")
st.code(f"explained_variance_ratio_ = {pca.explained_variance_ratio_}\ntotal_explained_variance = {np.sum(pca.explained_variance_ratio_)}", language="python")
st.markdown("""
    The first 3 principal components (PCs) explain almost 80% of the variance in the data, with the fist one alone explaining 45%, while the second and third components explain about 19% and 14% of the variance respectively. This means that we can use these 3 components to represent the data in a lower dimensional space without losing too much information.
            
    As mentioned before, a good thing about PCA is that it is easy to interpret, we can see which features are contributing the most to each component. This is done by looking at the loadings of each feature in each component. The loadings are the coefficients of the linear combination of the original features that make up each principal component. The higher the absolute value of the loading, the more important the feature is for that component.
            """)

fig4 = plot_pca_component_loadings(pca)
st.pyplot(fig4)
plt.close(fig4)

st.markdown("""
    Each of the components explains for a common kind of Pokémon.
    - **PC1**: Pokémon with overall balanced stats, with a bit more emphasis on attack and special attack, and a usually low speed.
    - **PC2**: Very fast and extremely weak Pokémon in terms of defense and hp.´
    - **PC3**: Big physical attackers with a good amount of speed but bad special attack or special defense.

    We commonly associate certain Pokemon types with certain stats, for example, we expect a rock type to be a bit slower but with a lot of attack and defense, while we expect a flying type to be fast but weak. Let's see if this kind of assumptions are true by plotting the Pokémon in the PCA space and coloring them by their main type.
            """)

fig5 = plot_pca(pokemon_data)
st.plotly_chart(fig5, use_container_width=True)

# Explain outliers
st.markdown("""
    Most of the Pokémon are well grouped in an ellipsoid shape, however, there is a few outliers, out of which 2 of them stand out as you can probably see. These Pokémon are *Shuckle* and *Eternatus Eternamax*. Let's see what they are doing there by looking at their stats.
""")
col1, col2 = st.columns(2)
with col1:
    st.image("imgs/shuckle.png", caption="Shuckle stats")
with col2:
    st.image("imgs/eternatus.png", caption="Eternatus stats")

st.markdown("""
    - Starting with *Shuckle*, this Pokémon has extremely low speed and attack, giving it very low values in the PC2 and PC3 axes. In the other hand, it has very high defense and special defense, which is what makes it mantain a positive value in the PC1 axis.      
    - *Eternatus Eternamax* is special case, as it is a special form of *Eternatus* that only appears as an opponent in the Pokémon Sword and Shield games story. It has the highest sum of stats of any Pokémon, with a total of 1125. This is what makes it stand out in the PCA space, as it has very high values in all the axes.
    
    Before moving on, let's see if the descriptions we made about each one of the PCs are true by looking at the representative Pokémon for each one of them. For this, we are going to find the Pokémon that are closest to the extremes of each one of the PCs.
    We are going to use the distance from the origin to find the closest Pokémon to the extremes of each one of the PCs. This is done by finding the Pokémon that are closest to the points (max, 0, 0), (0, max, 0) and (0, 0, max) in the PCA space. 
            """)

st.markdown("**Highest PC1 Pokémon**")
find_and_plot_k_nearest(pokemon_data, [pca_result[:, 0].max(), 0, 0], k=5)
st.markdown("""All the Pokémon in this list are legendary Pokémon, which is not surprising since they are usually very strong and have a lot of stats. These in particular are very well balanced Pokémon, with a good amount of all stats.""")

st.markdown("**Highest PC2 Pokémon**")
find_and_plot_k_nearest(pokemon_data, [0, pca_result[:, 1].max(), 0], k=5)
st.markdown("""Representing the PC2 axis, we have Pokémon that are very fast but weak in defense, confirming our assumption. *Accelgor* leads the list, which is well known for this characteristics. *Hisuian Electrode* seems to be just a reskinned *Electrode* in terms of stats.""")

st.markdown("**Highest PC3 Pokémon**")
find_and_plot_k_nearest(pokemon_data, [0, 0, pca_result[:, 2].max()], k=5)
st.markdown("""Lastly, we have the "glass cannon" Pokémon, which are very strong in attack but weak in special attack and special defense, which again is what we expected. This time we have 3 different variants of *Darmanitan* leading the list.""")

st.markdown("""
    The scatter plot is too noisy to get an idea of the distribution of the Pokémon of each type in the PCA space, so let's try to clean it up a bit to make things clearer.
    First, we are going to remove the outliers by calculating the centroid of each type, computing the distance of each Pokémon to the centroid and making a cutoff at the 80th percentile.
    Next, we are going to plot the convex hull of the Pokémon of each type, in other words, the smallest convex shape that contains all the Pokémon of each type.
    """)

fig6 = plot_convex_hull(pca_result, pokemon_data)
st.plotly_chart(fig6, use_container_width=True)

st.markdown("""
    Things are easier to see now, feel free to play around with the plot, activating and deactivating the different types to see how they are distributed in the PCA space.
    To save you some time, here are some interesting observations I made:
    """)


st.image("imgs/hull_flying_rock.png", caption="Convex hulls of Flying and Rock types in PCA space")
st.markdown("""
    Even when the Flying type is the most common one as we saw before, the distribution of the Pokémon in this type is very narrow so the possibilities are limited. Compare it to another type like Rock, whith Pokémon that differ a lot from each other in terms of stats.
        """)

st.image("imgs/hull_dragon_bug.png", caption="Convex hulls of Dragon and Bug types in PCA space")
st.markdown("""
    When you chose a cool looking Dragon Pokémon over the poor bug you found in route 1, you were almost certainly making a good choice. The distribution of these two types are pretty much disjoint, with the Dragon Type having much more powerful Pokémon than the Bug type, which is specially noticeable when looking at the PC1 axis.
        """)

st.image("imgs/hull_electric_ground.png", caption="Convex hulls of Electric and Ground types in PCA space")
st.markdown("""
    In a similar fashion to the previous example, there are other types that are very disjoint, such as Electric and Ground.
            In this case the choice is not as easy though. Both Electric and Ground types have a wide range of Pokémon in the PC1 axis, but generally speaking, Electric Pokémon are faster while Ground Pokémon have a better attack stat.
            """)


# st.header("4. Understanding competitive Pokémon teams")


st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 1.1em;'>
        Thank you for reading. Please keep in mind that this project was created as part of a personal challenge to learn more about data analysis and visualization and to have fun with Pokémon data. Some of the results might not be accurate or relevant, and I encourage you to do your own research and analysis.
        If you have any questions or suggestions, please feel free to contact me. I would love to hear your feedback and ideas for future projects!
        <br><br>
        <b>Author:</b> Juan Eizaguerri<br>
        <a href="https://github.com/jeizaguerri/pokemon-pca" target="_blank">Check out the code on GitHub</a><br>
    </div>
    """,
    unsafe_allow_html=True
)