import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
from urllib.request import urlopen
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
from PIL import Image
from statistics import median
import networkx as nx

st.set_page_config(page_title="ESP LaLiga Overview")


#mongodb connection
client = MongoClient(st.secrets['url_con'])
db = client.football_data
collection = db.fotmob_stats


#setting colors

background = '#e1ece1'
text = '#073d05'


#filtering data
YEAR = st.secrets['year']
LEAGUE = st.secrets['league']
IMAGE_LEAGUE = st.secrets['image_league']
GENTILIC = st.secrets['gentilic']

teams = list(collection.find({'general.league': LEAGUE, 'general.season': YEAR}).distinct('teams.home.name'))

#function collect all data
@st.cache_data(ttl='12h', show_spinner=False)
def get_data(league: str, season: str, country: str, teams: list) -> pd.DataFrame:

    
    squads = []
    passes_opp_half = []
    xg_for = []
    xg_diff = []
    url_images = []


    
    for team in teams:
        stats = collection.aggregate([{'$match': {'general.season': season, 'general.league': league, 'general.country': country, '$or': [{'teams.home.name': team}, {'teams.away.name': team}]}}, 
                                    {'$project': {'_id': 0, 'teams.home.name': 1, 'teams.away.name': 1, 'teams.home.image': 1, 'teams.away.image': 1, 'stats': 1}}])
        
        for stat in stats:
            
            squads.append(team)
            if stat['teams']['home']['name'] == team:
                passes_opp_half.append(stat['stats']['passes_opp_half_%']['home']*100)
                xg_for.append(stat['stats']['xg_op_for_100_passes']['home'])
                xg_diff.append(stat['stats']['xg_op_for_100_passes']['home'] - stat['stats']['xg_op_for_100_passes']['away'])
                url_images.append(stat['teams']['home']['image'])
                
            else:
                passes_opp_half.append(stat['stats']['passes_opp_half_%']['away']*100)
                xg_for.append(stat['stats']['xg_op_for_100_passes']['away'])
                xg_diff.append(stat['stats']['xg_op_for_100_passes']['away'] - stat['stats']['xg_op_for_100_passes']['home'])
                url_images.append(stat['teams']['away']['image'])
               

    #creating a dataframe
    df = pd.DataFrame({
        'Team': squads, 
        'Passes Opp Half %': passes_opp_half,
        'xG Open Play 100 Passes': xg_for, 
        'xG Open Play 100 Passes Diff': xg_diff, 
        'Image': url_images
    })

    df = df.groupby(by=['Team', 'Image']).mean()
    df = df.reset_index()
    df = df.round(2)

    return df

#function get images
@st.cache_data(ttl='30d', show_spinner=False)
def get_image(path):
    resp = urlopen(path)
    image = Image.open(resp)
    image = image.convert('RGBA')

    return OffsetImage(image, zoom=0.19, alpha=0.95)

#function get the team without re-run all app
@st.fragment
def get_team() -> None:

    if 'squad' not in st.session_state:
        st.session_state['squad'] = teams[0]

    squad = st.selectbox(label='Select a Squad', options=teams, index=0)
    st.session_state['squad'] = squad


    team_stats = collection.aggregate([{"$match": {"general.season": YEAR, "general.league": LEAGUE, 
                                    "$or": [{"teams.home.name": st.session_state['squad']}, {"teams.away.name": st.session_state['squad']}]}}, {"$project": {"_id": 0, "teams.home.name": 1, 
                                                                                                                                "teams.away.name": 1, 'stats.xg_op_for_100_passes.home': 1, 
                                                                                                                                "stats.xg_op_for_100_passes.away": 1, "formations.home": 1, 
                                                                                                                                "formations.away": 1}}])
    opponents = []
    xg_for = []
    xg_opp = []
    formation = []
    formation_opp = []

    for f in team_stats:
        if f['teams']['home']['name'] == squad:
            venue = 'home'
            venue_opp = 'away'
        else:
            venue = 'away'
            venue_opp = 'home'
            
        opponents.append(f['teams'][venue_opp]['name'])
        xg_for.append(f['stats']['xg_op_for_100_passes'][venue])
        xg_opp.append(f['stats']['xg_op_for_100_passes'][venue_opp])
        formation.append(f['formations'][venue])
        formation_opp.append(f"{f['formations'][venue_opp]}*")

    df2 = pd.DataFrame({
        "Opponent": opponents, 
        'xG Open Play Per 100 Passes For': xg_for, 
        'xG Open Play Per 100 Passes Opp': xg_opp, 
        'Formation': formation, 
        'Formation Opp': formation_opp
    })


    df_grouped = df2.groupby(by=['Formation', 'Formation Opp'])[['xG Open Play Per 100 Passes For', 'xG Open Play Per 100 Passes Opp']].mean().reset_index()

    G = nx.Graph()
    G.add_nodes_from(df_grouped['Formation'].unique(), nodetype='g')
    G.add_nodes_from(df_grouped['Formation Opp'].unique(), nodetype='r')
    colors = [u[1] for u in G.nodes(data='nodetype')]

    for index, row in df_grouped.iterrows():
        diff = row['xG Open Play Per 100 Passes For'] - row['xG Open Play Per 100 Passes Opp']
        G.add_edge(row['Formation'], row['Formation Opp'], diff=np.round(diff, 2))

    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["diff"] > 0]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["diff"] <= 0]

    fig2, ax2 = plt.subplots(figsize=(30, 12))

    degrees = nx.degree(G)
    pos = nx.kamada_kawai_layout(G)

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=[s[1]*800 for s in degrees],  node_color=colors, alpha=0.85, ax=ax2)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6, alpha=.5, edge_color='g', ax=ax2)
    nx.draw_networkx_edges(
        G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="r", style="dashed", ax=ax2
        
    )


    # node labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_weight='bold', labels={node: str(node).replace('*', '') for node in G.nodes}, ax=ax2)
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "diff")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,font_color='k', font_size=16, font_weight='bold', ax=ax2)

    fig2.patch.set_facecolor(background)
    ax2.patch.set_facecolor(background)

    df3 = df2.copy()
    df3['Formation Opp'] = df3['Formation Opp'].str.replace('*', '')

    st.dataframe(df3, hide_index=True)

    st.subheader(f"{st.session_state['squad']} Formations (Green) Vs Opponents Formations (Red)")
    st.write('This Graph is based on xG Open Play per 100 Passes Difference Average')

    st.pyplot(fig2)

with st.sidebar:
    st.image('static/image.jpeg', 
             caption="Saulo Faria - Data Scientist Specialized in Football")
    st.write("This App was designed in order to get an overview of Spanish LaLiga regarding xG Open Play Per 100 Passes")

    st.subheader("My links (pt-br)")
    st.link_button("Aposta Consciente", "https://apostaconsciente.hotmart.host/product-page-88be95cc-1892-4fa6-b364-69a271150f8f", use_container_width=True)
    st.link_button("Udemy", "https://www.udemy.com/user/saulo-faria-3/", use_container_width=True)
    st.link_button("Instagram", "https://www.instagram.com/saulo.foot/", use_container_width=True)
    st.link_button("X", "https://x.com/fariasaulo_", use_container_width=True)
    st.link_button("Youtube", "https://www.youtube.com/channel/UCkSw2eyetrr8TByFis0Uyug", use_container_width=True)

st.html("""
        <style>
            .stMainBlockContainer {
                max-width:90rem;
            }
        </style>
        """
    )
col1, col2 = st.columns([7, 1], vertical_alignment='center')

with col1:
    st.title(f'xG Open Play Per 100 Passes Charts - {GENTILIC} {LEAGUE} {YEAR}')
with col2:
    st.image(IMAGE_LEAGUE, width=100)

#get data
df = get_data(league=LEAGUE, season=YEAR, country='ESP', teams=teams)
df_plot = df[['Team', 'Passes Opp Half %', 'xG Open Play 100 Passes', 'xG Open Play 100 Passes Diff']]
df_styled = df_plot.style.background_gradient(cmap='Greens', text_color_threshold=0.5, 
                                                                subset=['Passes Opp Half %', 'xG Open Play 100 Passes'], low=0.1).format(precision=2).background_gradient(cmap='RdYlGn', 
                                                                                                                        text_color_threshold=0.5, 
                                                                                                                        subset=['xG Open Play 100 Passes Diff'], low=0).format(precision=2)

    
#plot 1
fig, ax = plt.subplots(figsize=(20, 8))
ax.scatter(df['xG Open Play 100 Passes'], df['xG Open Play 100 Passes Diff'], c=background)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('k')
ax.spines['left'].set_color('k')
ax.set_xlabel('xG Open Play Per 100 Passes', fontweight='bold', fontsize=12, color=text)
ax.set_ylabel('xG Open Play Per 100 Passes Diff', fontweight='bold', fontsize=12, color=text)
ax.tick_params(
    axis='both', 
    which='both', 
    colors=text,
)

fig.patch.set_facecolor(background)
ax.patch.set_facecolor(background)

for index, row in df.iterrows():    
    ab = AnnotationBbox(get_image(row['Image']), (row['xG Open Play 100 Passes'], row['xG Open Play 100 Passes Diff']), frameon=False)
    ax.add_artist(ab)

    ax.hlines(df['xG Open Play 100 Passes Diff'].median(), df['xG Open Play 100 Passes'].min(), df['xG Open Play 100 Passes'].max(), 
            color=text, alpha=0.3)
    ax.hlines(np.percentile(df['xG Open Play 100 Passes Diff'], 25), df['xG Open Play 100 Passes'].min(), df['xG Open Play 100 Passes'].max(), 
            color=text, alpha=0.1, **{'ls': '--'})
    ax.hlines(np.percentile(df['xG Open Play 100 Passes Diff'], 75), df['xG Open Play 100 Passes'].min(), df['xG Open Play 100 Passes'].max(), 
            color=text, alpha=0.1, **{'ls': '--'})


    ax.vlines(df['xG Open Play 100 Passes'].median(), df['xG Open Play 100 Passes Diff'].min(), df['xG Open Play 100 Passes Diff'].max(), 
            color=text, alpha=0.3)
    ax.vlines(np.percentile(df['xG Open Play 100 Passes'], 25), df['xG Open Play 100 Passes Diff'].min(), df['xG Open Play 100 Passes Diff'].max(), 
            color=text, alpha=0.1, **{'ls': '--'})
    ax.vlines(np.percentile(df['xG Open Play 100 Passes'], 75), df['xG Open Play 100 Passes Diff'].min(), df['xG Open Play 100 Passes Diff'].max(), 
            color=text, alpha=0.1, **{'ls': '--'})

            
#plot2
fig2, ax2 = plt.subplots(figsize=(20, 8))
ax2.scatter(df['Passes Opp Half %'], df['xG Open Play 100 Passes Diff'], c=background)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_color('k')
ax2.spines['left'].set_color('k')
ax2.set_xlabel('Passes Opp Half %', fontweight='bold', fontsize=12, color=text)
ax2.set_ylabel('xG Open Play Per 100 Passes Diff', fontweight='bold', fontsize=12, color=text)
ax2.tick_params(
    axis='both', 
    which='both', 
    colors=text,
)

fig2.patch.set_facecolor(background)
ax2.patch.set_facecolor(background)

for index, row in df.iterrows():    
    ab = AnnotationBbox(get_image(row['Image']), (row['Passes Opp Half %'], row['xG Open Play 100 Passes Diff']), frameon=False)
    ax2.add_artist(ab)

    ax2.hlines(df['xG Open Play 100 Passes Diff'].median(), df['Passes Opp Half %'].min(), df['Passes Opp Half %'].max(), 
            color=text, alpha=0.3)
    ax2.hlines(np.percentile(df['xG Open Play 100 Passes Diff'], 25), df['Passes Opp Half %'].min(), df['Passes Opp Half %'].max(), 
            color=text, alpha=0.1, **{'ls': '--'})
    ax2.hlines(np.percentile(df['xG Open Play 100 Passes Diff'], 75), df['Passes Opp Half %'].min(), df['Passes Opp Half %'].max(), 
            color=text, alpha=0.1, **{'ls': '--'})


    ax2.vlines(df['Passes Opp Half %'].median(), df['xG Open Play 100 Passes Diff'].min(), df['xG Open Play 100 Passes Diff'].max(), 
            color=text, alpha=0.3)
    ax2.vlines(np.percentile(df['Passes Opp Half %'], 25), df['xG Open Play 100 Passes Diff'].min(), df['xG Open Play 100 Passes Diff'].max(), 
            color=text, alpha=0.1, **{'ls': '--'})
    ax2.vlines(np.percentile(df['Passes Opp Half %'], 75), df['xG Open Play 100 Passes Diff'].min(), df['xG Open Play 100 Passes Diff'].max(), 
            color=text, alpha=0.1, **{'ls': '--'})

st.dataframe(df_styled, hide_index=True)
st.divider()

tab1, tab2 = st.tabs(['xG For x Diff', 'Passes Opp Half x xG Diff'])

with tab1:
    st.subheader('xG Open Play Per 100 Passes For x Diff')
    st.pyplot(fig)
with tab2:
    st.subheader('Passes Opp Half % x xG Open Play 100 Passes Diff')
    st.pyplot(fig2)


st.divider()

get_team()

st.caption("This App Was Developed by Saulo Faria - Data Scientist Specialized in Football (Soccer)")
