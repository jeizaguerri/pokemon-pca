import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


from visualization import *
from app_logic import *

os.chdir(os.path.dirname(os.path.abspath(__file__)))

INPUT_DATA_FILE = 'pokemon_data.csv'
STAT_ROWS = ['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']
TEAM_SIZE = 6

st.set_page_config(layout="centered")  # Ensure proper layout

st.title("Analyzing Pokémon and Building Competitive Teams in a Low Dimensional Space")
st.subheader("By Juan Eizaguerri")
st.image("imgs/banner.png")
st.markdown("""
    Visualizing data in a graphical way often helps to understand the data better, however, this is not easy to do when dealing with high dimensional data.
    I wanted to put this into practice and thought that it would be a good idea to use data I am already familiar with, given that I should be able to tell when a result is wrong or unexpected.
    This is how I ended up with a needlessly complex (but fun) analysis on Pokémon stats and types. *Are there type combinations yet to be explored? Could they be classified in large groups? Do Pokémon stats correlate with their types? Could we use this information to build a better team?* I will try to answer these questions in this project.
    """)

st.info("""
    NOTE: Most of the plots and results here are calculated in real time to be fully interactive. This means some of the plots might take some time to load and you could encounter some errors. If you do, please feel free to contact me and I will try to fix it as soon as possible.
    """)

st.header("1. Input data")
st.markdown("""
    Before starting to process and analyze the data, we first need to load it from somewhere. Thankfully, the people from [PokéAPI](https://pokeapi.co/) are doing an exceptional job at providing Pokémon data in a structured way.
    A few simple requests to their API and we can get all the data we need. For this project, I only wanted to get the Pokémon stats and types, their names, and their sprite for visualization purposes, although arguably, other data such as abilities, moves, and evolutions could be interesting too to answer some of the questions I proposed in the introduction.
    It wouldn't be too polite to request all the data every time we want to analyze it, so I decided to save the data in a [CSV](https://github.com/jeizaguerri/pokemon-pca/blob/main/pokemon_data.csv) file which you can find in the directory this app. You can also check the code I used to get the data [here](https://github.com/jeizaguerri/pokemon-pca/blob/main/load_stats.py).
            """)

if not os.path.exists(INPUT_DATA_FILE):
    st.error(f"Input data file '{INPUT_DATA_FILE}' not found.")
    st.stop()

st.success("Data file found!")
try:
    pokemon_data = pd.read_csv(INPUT_DATA_FILE)
    st.write("Here is a little preview of the data:")
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
    This is important to keep in mind when analyzing the data, as we are going to be filtering the data by type in future sections. In these cases we will only be using the first type, as the second type is not always available.
    
    Next, let's see how many Pokémon there are of each type (having main and secondary type in mind), which will be useful to understand the distribution of types in the data and to see if there are any types that are over or under represented.
    """)

type_counts = count_pokemon_by_type(pokemon_data)
fig2 = plot_pokemon_by_type(type_counts)
st.pyplot(fig2)
plt.close(fig2)
st.markdown("""
    While there is a good amount of Pokémon of each type, it is surprising to see such an unbalanced distribution. It makes sense that the later introduced types such as fairy have less Pokémon, but it is interesting to see that some types such as ice, which is a type that was introduced in the first generation, have so few Pokémon, global warming seems to be affecting Pokémon too.

    Let's dig a bit deeper and see if there are more common combinations of types, or if there are types combinations that are yet to be explored. For this, we are going to create a co-occurrence matrix of types. The upper triangle of the matrix will be redundant so we will only show the lower triangle, the main diagonal is also removed since there are no Pokémon with the same type twice.
            """)

type_cooccurrence = get_type_cooccurrence(pokemon_data)
fig3 = plot_type_cooccurrence(type_cooccurrence)
st.pyplot(fig3)
plt.close(fig3)

st.markdown("""
    Good news for normal-flying fans, you can fill a full box of Pokémon with this type combination in your [PC](https://bulbapedia.bulbagarden.net/wiki/Pok%C3%A9mon_Storage_System#Limitations). Overall, most of the combinations are covered, but there are exactly 9 combinations that are yet to be explored. Here is the full list:
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
        
st.header("3. Performing Analysis in PCA Space")
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
    - **PC2**: Very fast and extremely weak Pokémon in terms of defense and hp.
    - **PC3**: Big physical attackers with a good amount of speed but bad special attack or special defense.

    We commonly associate certain Pokemon types with certain stats, for example, we expect a rock type to be a bit slower but with a lot of attack and defense, while we expect a flying type to be fast but weak. Let's see if this kind of assumptions are true by plotting the Pokémon in the PCA space and coloring them by their main type.
            """)

fig5 = plot_pca(pokemon_data)
st.plotly_chart(fig5, use_container_width=True)

# Explain outliers
st.markdown("""
    Most Pokémon are well grouped in an ellipsoid shape, however, there is a few outliers, out of which 2 of them stand out as you can probably see. These Pokémon are *Shuckle* and *Eternatus Eternamax*. Let's see what they are doing there by looking at their stats.
""")
col1, col2 = st.columns(2)
with col1:
    st.image("imgs/shuckle.png", caption="Shuckle stats")
with col2:
    st.image("imgs/eternatus.png", caption="Eternatus stats")

st.markdown("""
    - Starting with *Shuckle*, this little guy has extremely low speed and attack, giving it very low values in the PC2 and PC3 axes. In the other hand, it has very high defense and special defense, which is what makes it mantain a positive value in the PC1 axis.      
    - *Eternatus Eternamax* is special case, as it is a special form of *Eternatus* that only appears as an opponent in the Pokémon Sword and Shield games story. It has the highest sum of stats of any Pokémon, with a total of 1125. This is what makes it stand out in the PCA space, as it has very high values in all the axes.
    
    Before moving on, let's see if the descriptions we made about each one of the PCs are true by looking at the representative Pokémon for each one of them. For this, we are going to find the Pokémon that are closest to the extremes of each one of the PCs.
            This is done by finding the Pokémon that are closest to the points (max, 0, 0), (0, max, 0) and (0, 0, max) in the PCA space. 
            """)

st.markdown("**Highest PC1 Pokémon**")
find_and_plot_k_nearest(pokemon_data, [pca_result[:, 0].max(), 0, 0], k=5)
st.markdown("""All the Pokémon in this list are legendary Pokémon, which is not surprising since they are usually very strong and have a lot of stats. These in particular are very well balanced Pokémon, with a good amount of all stats.""")

st.markdown("**Highest PC2 Pokémon**")
find_and_plot_k_nearest(pokemon_data, [0, pca_result[:, 1].max(), 0], k=5)
st.markdown("""Representing the PC2 axis, we have Pokémon that are very fast but weak in defense, confirming our assumption. *Accelgor* leads the list, which is well known for this characteristics. *Hisuian Electrode* seems to be just a reskinned *Electrode* in terms of stats.""")

st.markdown("**Highest PC3 Pokémon**")
find_and_plot_k_nearest(pokemon_data, [0, 0, pca_result[:, 2].max()], k=5)
st.markdown("""Lastly, we have the "glass cannon" Pokémon, all of them are very strong in attack but weak in special attack and special defense, which again is what we expected. This time we have 3 different variants of *Darmanitan* leading the list.""")

st.markdown("""
    The scatter plot is too noisy to get an idea of the distribution of the Pokémon of each type in the PCA space, so let's try to clean it up a bit to make things clearer.
    First, we are going to remove the outliers by calculating the centroid of each type, computing the euclidean distance of each Pokémon to the centroid and making a cutoff at the 80th percentile.
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
    Even when the Flying type is the most common one as we saw before, the distribution of the Pokémon in this type is very narrow so the possibilities are limited. Compare it to another type like Rock, with Pokémon that differ a lot from each other in terms of stats.
        """)

st.image("imgs/hull_dragon_bug.png", caption="Convex hulls of Dragon and Bug types in PCA space")
st.markdown("""
    When you chose a cool looking Dragon Pokémon over the poor bug you found in route 1, you were almost certainly making a good choice. The distribution of these two types are pretty much disjoint, with the Dragon Type having much more powerful Pokémon than the Bug type, which is especially noticeable when looking at the PC1 axis.
        """)

st.image("imgs/hull_electric_ground.png", caption="Convex hulls of Electric and Ground types in PCA space")
st.markdown("""
    In a similar fashion to the previous example, there are other types that are very disjoint, such as Electric and Ground.
            In this case the choice is not as easy though. Both Electric and Ground types have a wide range of Pokémon in the PC1 axis, but generally speaking, Electric Pokémon are faster while Ground Pokémon have a better attack stat.
            """)


st.header("4. Understanding the competitive Pokémon Meta")
st.markdown("""
    Even though there are a lot of different Pokémon to choose from when playing through the games, the competitive scene is usually dominated by a few Pokémon that are considered to be the best in the game, the ones that set the meta.
    The popularity of a Pokémon in the competitive scene is not only determined by its stats, but also by its typing, movepool, abilities and synergy with other Pokémon, but it is still an important factor to consider, so it would make sense to find some correlation between the stats of a Pokémon and its usage in the competitive scene.
    Let's find out where the most popular Pokémon are in the PCA space and see if we can extract any information from it.
    
    We are going to be using the data provided by [Smogon](https://www.smogon.com/stats/) which is a competitive Pokémon community specializing in the art of competitive battling. They provide a lot of information extracted from the [Pokémon Showdown simulator](https://play.pokemonshowdown.com/).
    They publish reports every month for different tiers and modes. We will be using the data from december 2024, OU Tier, which is the most popular tier in the game, related to the most popular Pokémon. Also we will only be using the data from players with an elo rating of 1825 or higher. You can find the raw data [here](https://www.smogon.com/stats/2024-12/gen9ou-1825.txt).
    This information is not provided in a structured manner, but rather as a plain text file, so a bit of regex magic is needed to extract the different fields that we are looking for. I wrote a small script to do this, which you can find [here](https://github.com/jeizaguerri/pokemon-pca/blob/main/load_competitive.py). As we did for the stats data, we will save the processed data in a [CSV](https://github.com/jeizaguerri/pokemon-pca/blob/main/smogon_usage_stats.csv) file to avoid having to process it every time.
    """)
if not os.path.exists('smogon_usage_stats.csv'):
    st.error("Smogon usage stats file 'smogon_usage_stats.csv' not found.")
    st.stop()
st.success("Smogon usage stats file found!")
try:
    smogon_data = pd.read_csv('smogon_usage_stats.csv')
    st.write("You can preview the data here:")
    st.dataframe(smogon_data)
except Exception as e:
    st.error(f"Failed to load data: {e}")

st.markdown("""
    As you can see, the table shows the usage of each Pokémon in both raw and percentage values. The *real* column can be a bit misleading, as it means the total times this Pokémon appeared in battle / total number of pokemon actually sent out in battle.
    What we are interested in is the *usage* column, which shows the percentage of battles in which the Pokémon was used. This means that if a Pokémon has a usage of 10%, it was used in 10% of the battles.
            
    All that is left to do is to combine this data with the stats data we have, so we can see how the most popular Pokémon are distributed in the PCA space. We will do this by merging the two dataframes using the name of the Pokémon as the key. This will require some processing since the names are not exactly the same in both dataframes, but I will save you the trouble so that we can jump straight to the fun part.
            """)

# Merge the dataframes
pokemon_data = merge_dataframes(pokemon_data, smogon_data)
with st.expander("Show merged Pokémon + Smogon data"):
    st.dataframe(pokemon_data)

st.markdown("""
    With this done, let's repeat the scatter plot we did before, but this time we will color the Pokémon by their usage in the competitive scene.
    We will also add a slider to filter the Pokémon by their usage, so we can focus on most popular Pokémon if needed.
            """)

# Filter the Pokémon by their usage
usage_filter = st.slider(
    "Select the usage percentage range:",
    min_value=0.0,
    max_value=100.0,
    value=(0.0, 100.0),
    step=1.0,
)
filtered_pokemon_data = pokemon_data[
    (pokemon_data['usage_percent'] >= usage_filter[0]) &
    (pokemon_data['usage_percent'] <= usage_filter[1])
]
st.markdown(f"Showing Pokémon with usage between {usage_filter[0]}% and {usage_filter[1]}%")

# Plot the filtered Pokémon
fig7 = plot_pca_usage(filtered_pokemon_data)
st.plotly_chart(fig7, use_container_width=True)
st.markdown("""
    PCA1 must be a good indicator of a Pokémon viability in the competitive scene, as there is no Pokémon with negative values in this axis with a usage of more than 5%.
    The most used Pokémon conform an outer shell around the less used Pokémon in the PCA space. The ones that escape from this shell are not allowed in this competitive league, most of them being legendaries and mega evolutions.
            """)

st.header("5. Using GMMs and MMO to generate Pokémon teams")
st.markdown("""
    Now that we have seen that stats DO matter when it comes to building a competitive team, nothing is stopping us from using this information to automatically build viable teams.
            
    1. We first need to train a model to learn the distribution of the best Pokémon (Usage over 1%) in the PCA space. The shape of the data is not too complex, so we can use a simple [Gaussian Mixture Model (GMM)](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model) to do this. This model represents the data distribution as a weighted sum of multiple Gaussian distributions, each capturing a different cluster or mode in the data.
    2. Once we have trained the model, we can sample from it to generate new Pokémon with similar stats to the ones in the training set. This is done by sampling from each Gaussian distribution in the mixture according to its weight.
    3. The points generated by the model will not match any real Pokémon, so the next step is to find the closest Pokémon to each generated point. This is done by calculating the euclidean distance from the generated point to each Pokémon in the training set and selecting the closest one.
    
    GMMs only have a single hyperparameter, which is the number of components to use. This is a bit tricky to choose, as we don't know how many clusters there are in the data. A good way to do this is to use the [Bayesian Information Criterion (BIC)](https://en.wikipedia.org/wiki/Bayesian_information_criterion), which is a measure of the goodness of fit of the model. The lower the BIC, the better the model fits the data.
    We will fit models to our data with different number of components between 1 and 15 and select the one with the lowest BIC.
            """)

filtered_pokemon_data = pokemon_data[pokemon_data['usage_percent'] >= 1.0]
components = range(1, 16)
bic_values = test_bics(components, filtered_pokemon_data)
fig8 = plot_bic(components, bic_values)
st.plotly_chart(fig8, use_container_width=True)

st.markdown("""
    Okay, this was unexpected. It seems like a single Gaussian distribution is enough to represent the data. This is not too surprising when thinking about it, as we saw that the data was well grouped in a single ellipsoid shape. I am not complaining since this will make the execution of the model even faster.
    This should be enough to generate individual competitive Pokémon. Let's see how this works in practice by generating a few Pokémon.
            """)

gmm = train_gmm(filtered_pokemon_data, n_components=1)
generated_pokemon = generate_pokemon(gmm, samples=10, pokemon_data=pokemon_data)
st.markdown("<h3 style='text-align: center;'>Generated Pokémon</h2>", unsafe_allow_html=True)
display_generated_pokemon(generated_pokemon, n_columns=5)
col_center = st.columns([1, 2, 1])
with col_center[1]:
    if st.button("Regenerate Pokémon", use_container_width=True, icon="⚡"):
        gmm = train_gmm(filtered_pokemon_data, n_components=1)
        generated_pokemon = generate_pokemon(gmm, samples=10, pokemon_data=pokemon_data)




# IMPLEMENTATION

st.markdown("""
    The model seems to be suggesting good Pokémon. The last step in our journey is to use this generator to build teams of 6 Pokémon. As you probably already thought, generating 6 individually viable Pokémon is not enough to build a team. We need to make sure that the Pokémon we generate are not only good on their own, but also work well together as a team.
    I initially wanted to do this by some kind of supervised learning trained on popular team compositions, but sadly I couldn't find a good dataset to train the model. Instead, I decided to use a more heuristic approach, converting the problem into a [multi-objective optimization (MOO) problem](https://en.wikipedia.org/wiki/Multi-objective_optimization).
    This kind of problems involve finding solutions that balance two or more conflicting objectives while satisfying a set of constraints. Instead of a single best answer, MOO seeks a set of [Pareto optimal solutions](https://es.wikipedia.org/wiki/Eficiencia_de_Pareto).

    Applying this to pokémon team building is an extremely hard task as there are a lot of factors to consider, we are going to simplify the problem a bit by only considering the following objectives:
    - **Raw stats**: We saw earlier that stats usually correlate with usage, so it makes sense to use them as a metric to evaluate the teams. We are going to use the sum of all the stats of the Pokémon in the team as a metric to evaluate the teams. The higher the sum, the better the team is.
    - **Coverage**: The team should cover as many types as possible. This is important to avoid having a single type weakness, which can be exploited by the opponent. This is easy to calculate by simply counting the number of different types in the team.
    - **Balance**: The team should be balanced in terms of stats. This is important to make sure we have a good mix of offensive and defensive Pokémon, and don't end up with a team full of glass cannons or tanks. For this, we are going to use the average pair distance between the Pokémon in the team in the PCA space. The higher the distance, the more balanced the team is.
       
    There is multiple ways to solve MOO problems, such as [genetic algorithms]() and [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning). Instead, we are going to use a simple grid search with Pareto filtering since it will be easier to implement and understand.
    All we need to do is to generate a lot of teams and then filter the ones that are Pareto optimal. This is done by comparing each team with all the other teams and checking if it is better in at least one objective and not worse in any other objective.
            """)

st.markdown("### Generate optimized teams")
n_teams = st.slider("Number of candidate teams to generate", 100, 3000, 1000, step=100)
usage_threshold = 0.0

if st.button("Generate Teams", use_container_width=True, icon="🛡️"):
    with st.spinner("Generating and evaluating teams..."):
        teams = generate_candidate_teams(pokemon_data, gmm, n_teams=n_teams, team_size=TEAM_SIZE, usage_threshold=0.01)
        team_scores = [evaluate_team(team) for team in teams]
        pareto_front = select_pareto_front(team_scores)
        best_teams = [teams[i] for i in pareto_front]
        best_scores = [team_scores[i] for i in pareto_front]

    st.success(f"Found {len(best_teams)} Pareto-optimal teams!")
    with st.expander("Show Pareto-optimal teams"):
        show_teams(best_teams, best_scores)
    
st.markdown("""
    And there you have it! I wouldn't trust this teams to try to win a tournament, but they look quite good on paper.
    This method would easily extensible by adding more objectives, such as synergy between Pokémon and coverage against common meta threads.
    It could even be used as a "auto-complete" feature, where you could select a few Pokémon and the app would generate a team for you.
            """)

st.header("6. Conclusion")
st.markdown("""
    I consider this project a success, as I was able to carry out all the ideas I had in mind when I started it. After some basic data analysis on the Pokémon types, I was able to perform PCA on the stats data and visualize all the Pokémon in a way that is easy to understand.
    This low-dimensional representation of the data allowed me to train a competitive Pokémon generator using a Gaussian Mixture Model. This model was later used to generate whole teams through a multi-objective optimization approach.
    There is still a lot of things to explore and improve, so feel free to take the [code for this project](https://github.com/jeizaguerri/pokemon-pca) and use it as a starting point for your own ideas.

    I hope you enjoyed this project as much as I did. I learned a lot about data analysis and visualization, and I had a lot of fun along the way.
    I also learned a bit more about Pokémon, which is always a plus. 
            """)

st.divider()
st.markdown(
    """
    <div style='text-align: center; font-size: 1.1em; color: #666666;'>
        Thank you for reading. Please keep in mind that this project was created as part of a personal challenge to learn more about data analysis and visualization and to have fun with Pokémon data. Some of the results might not be accurate or relevant, and I encourage you to do your own research and analysis.
        If you have any questions or suggestions, please feel free to contact me. I would love to hear your feedback and ideas for future projects!
        <br><br>
        <b>Author:</b> Juan Eizaguerri<br>
        <a href="https://github.com/jeizaguerri/pokemon-pca" target="_blank">Check out the code on GitHub</a><br>
    </div>
    """,
    unsafe_allow_html=True
)