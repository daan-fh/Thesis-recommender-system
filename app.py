import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pydeck as pdk

# --- Custom Styling ---
st.markdown("""
    <style>
        .stApp {
            background-color: white;
            color: black;
        }
        header[data-testid="stHeader"] {
            background-color: #2e2d5c;
            color: white;
        }
        header[data-testid="stHeader"] .stMarkdown {
            color: white;
        }
        section[data-testid="stSidebar"] {
            background-color: #2e2d5c;
            color: white;
        }
        section[data-testid="stSidebar"] * {
            color: white !important;
        }
        div[data-testid="stSidebar"] > button {
            color: white;
        }
        div[data-testid="stButton"] > button {
            background-color: #f9f9f9;
            color: black;
            border: 1px solid #ccc;
            border-radius: 10px;
            padding: 0.5em 1em;
            transition: all 0.2s ease-in-out;
        }
        div[data-testid="stButton"] > button:hover {
            background-color: #eaeaea;
            color: black;
            border: 2px solid #00b0a7;
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load and Prepare Data ---
df_project = pd.read_csv('csv/project.csv')
df_building_type = pd.read_csv('csv/project_building_type.csv')
df_building_type_text = pd.read_csv('csv/project_building_type_text.csv')

df_project = df_project[['id', 'address_city', 'living_space_from', 'living_space_to', 'sale_price_from', 'sale_price_to', 'geo_location']]
df_building_type = df_building_type[['id', 'project_id', 'bedrooms']]
df_building_type_text = df_building_type_text[['project_building_type_id', 'offer']]

df_project['id'] = pd.to_numeric(df_project['id'], errors='coerce').astype('Int64')
df_building_type['id'] = pd.to_numeric(df_building_type['id'], errors='coerce').astype('Int64')
df_building_type['project_id'] = pd.to_numeric(df_building_type['project_id'], errors='coerce').astype('Int64')
df_building_type_text['project_building_type_id'] = pd.to_numeric(df_building_type_text['project_building_type_id'], errors='coerce').astype('Int64')

merged_df = (
    df_project
    .merge(df_building_type, left_on='id', right_on='project_id', how='left')
    .merge(df_building_type_text, left_on='id_y', right_on='project_building_type_id', how='left')
)
merged_df = merged_df.dropna()
merged_df = merged_df[merged_df['geo_location'].str.count(',') == 1].copy()
merged_df = merged_df.drop(columns=['id_y', 'project_id', 'project_building_type_id'])
merged_df = merged_df.rename(columns={'id_x': 'id'})
merged_df = merged_df.reset_index(drop=True)

merged_temp_df = merged_df.copy()
merged_temp_df[['lat', 'lon']] = merged_temp_df['geo_location'].str.split(',', expand=True).astype(float)

merged_df = merged_df.drop_duplicates(subset=['geo_location'])
merged_temp_df = merged_temp_df.loc[merged_df.index]
merged_temp_df = merged_temp_df.reset_index(drop=True)
merged_df = merged_df.reset_index(drop=True)

features = ['living_space_from', 'living_space_to', 'sale_price_from', 'sale_price_to', 'bedrooms']
scaler = StandardScaler()
merged_temp_df[features] = scaler.fit_transform(merged_temp_df[features])

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(merged_df['offer'])

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def calculate_combined_distance(index, df, weights):
    target_row = df.iloc[index]
    distances = []
    for i in range(len(df)):
        if i == index:
            continue
        row = df.iloc[i]
        feature_vector_1 = target_row[features].values
        feature_vector_2 = row[features].values

        price_dist = np.linalg.norm(feature_vector_1[2:4] - feature_vector_2[2:4])
        size_dist = np.linalg.norm(feature_vector_1[0:2] - feature_vector_2[0:2])
        bed_dist = np.abs(feature_vector_1[4] - feature_vector_2[4])
        geo_dist = haversine_distance(target_row['lat'], target_row['lon'], row['lat'], row['lon'])
        text_sim = cosine_similarity(tfidf_matrix[index], tfidf_matrix[i])[0][0]
        text_dist = 1 - text_sim

        total_score = (
            weights['price'] * price_dist +
            weights['size'] * size_dist +
            weights['bedrooms'] * bed_dist +
            weights['geo'] * geo_dist +
            weights['text'] * text_dist
        )
        distances.append((i, total_score, price_dist, size_dist, bed_dist, geo_dist, text_dist))
    distances.sort(key=lambda x: x[1])
    return distances[:4]

def calculate_item_coverage(df, weights):
    total_items = len(df)
    all_recommended = set()

    for idx in range(total_items):
        recs = calculate_combined_distance(idx, df, weights)
        all_recommended.update([r[0] for r in recs])

    coverage = len(all_recommended) / total_items
    return coverage

# --- Streamlit UI ---
st.image("images/logo-nwn.png", width=160)
st.title("Aanbevelingssysteem voor Nieuw Wonen Nederland")

st.markdown("""
<h4>Welkom!</h4>

<p>Op deze pagina zie je alle beschikbare projecten in één overzicht.<br>
Klik op <strong>"Bekijk details"</strong> om meer informatie over een project te bekijken.</p>

<h5>Op de detailpagina zie je:</h5>
<ul>
    <li>Een duidelijke omschrijving van het project</li>
    <li>Vergelijkbare projecten die je mogelijk ook interessant vindt</li>
    <li>Hoe beter het project past bij wat je bekijkt, hoe lager de score. Dat is dus goed!</li>
</ul>

<p>Wil je terug naar het overzicht? Klik dan bovenaan op <strong>"Terug naar startpagina"</strong>.</p>

<p>Links bovenin zie je een klein pijltje. Daarmee open je een menu waarin je kunt aangeven wat jij belangrijk vindt (zoals prijs, grootte of locatie).
    Zo worden de suggesties nog beter afgestemd op jouw voorkeur.</p>

<p>Neem gerust even de tijd om te kijken of de voorgestelde projecten aansluiten bij wat je zoekt.</p>
""", unsafe_allow_html=True)


# Weight sliders
st.sidebar.header("Voorkeuren voor Aanbeveling")
price_weight = st.sidebar.slider("Gewicht prijs", 0.0, 1.0, 0.3)
size_weight = st.sidebar.slider("Gewicht woonoppervlakte", 0.0, 1.0, 0.3)
bedroom_weight = st.sidebar.slider("Gewicht aantal slaapkamers", 0.0, 1.0, 0.1)
geo_weight = st.sidebar.slider("Gewicht locatie", 0.0, 1.0, 0.2)
text_weight = st.sidebar.slider("Gewicht tekstuele overeenkomst", 0.0, 1.0, 0.1)

weights = {
    'price': price_weight,
    'size': size_weight,
    'bedrooms': bedroom_weight,
    'geo': geo_weight,
    'text': text_weight,
}

"""with st.sidebar:
    st.markdown("---")
    st.markdown("### Systeemdekking (Item Coverage)")
    with st.spinner("Berekenen..."):
        coverage = calculate_item_coverage(merged_temp_df, weights)
    st.metric("Item Coverage", f"{coverage:.2%}")"""

if 'selected_index' not in st.session_state:
    st.session_state.selected_index = None

def select_property(index):
    st.session_state.selected_index = index

def render_card(row, index, score_data=None):
    city = f"{row['address_city']} {index + 1}"
    price = f"{int(row['sale_price_from'])} - {int(row['sale_price_to'])} €"
    space = f"{int(row['living_space_from'])} - {int(row['living_space_to'])} m²"
    bedrooms = f"{int(row['bedrooms'])} slaapkamers"

    st.markdown(f"""
    <div style='border: 1px solid #ccc; border-radius: 10px; padding: 15px; margin-bottom: 10px; background-color: #f9f9f9;'>
        <strong>{city}</strong><br>
        Prijs: {price}<br>
        Oppervlakte: {space}<br>
        Slaapkamers: {bedrooms}<br>
    </div>
    """, unsafe_allow_html=True)

    if score_data:
        st.markdown(f"""
        **Score**: {score_data[1]:.2f}  
        - Prijsverschil: {score_data[2]:.2f}  
        - Oppervlakteverschil: {score_data[3]:.2f}  
        - Verschil slaapkamers: {score_data[4]:.2f}  
        - Afstand (km): {score_data[5]:.2f}  
        - Tekstverschil: {score_data[6]:.2f}
        """)

    st.button("Bekijk details", key=f"select_{index}", on_click=lambda: select_property(index))


if st.session_state.selected_index is None:
    st.subheader("Selecteer een woning om details te bekijken")
    cols = st.columns(2)
    for i, row in merged_df.iterrows():
        with cols[i % 2]:
            render_card(row, i)
else:
    selected_index = st.session_state.selected_index
    selected_row = merged_df.iloc[selected_index]

    if st.button("🔙 Terug naar overzicht"):
        st.session_state.selected_index = None
        st.rerun()

    st.subheader("Woningdetails")
    st.markdown(f"""
    **Stad**: {selected_row['address_city']}  
    **Prijs**: {int(selected_row['sale_price_from'])} - {int(selected_row['sale_price_to'])} €  
    **Woonoppervlakte**: {int(selected_row['living_space_from'])} - {int(selected_row['living_space_to'])} m²  
    **Slaapkamers**: {int(selected_row['bedrooms'])}  
    **Beschrijving**:  
    {selected_row['offer']}
    """, unsafe_allow_html=True)

    recommended = calculate_combined_distance(selected_index, merged_temp_df, weights)
    st.markdown("### Aanbevolen Woningen")
    rec_cols = st.columns(2)
    for i, (rec_idx, *score_data) in enumerate(recommended):
        with rec_cols[i % 2]:
            render_card(merged_df.iloc[rec_idx], rec_idx, score_data=(rec_idx, *score_data))

    map_df = pd.concat([
        merged_df.iloc[[selected_index]].assign(type='Geselecteerd'),
        merged_df.iloc[[r[0] for r in recommended]].assign(type='Aanbevolen')
    ])
    map_df[['lat', 'lon']] = map_df['geo_location'].str.split(',', expand=True).astype(float)

    st.markdown("### Kaart met Geselecteerde en Aanbevolen Woningen")
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=map_df['lat'].mean(),
            longitude=map_df['lon'].mean(),
            zoom=11,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position='[lon, lat]',
                get_color='[0, 128, 255]' if map_df.iloc[0]['type'] == 'Geselecteerd' else '[255, 0, 128]',
                get_radius=100,
                pickable=True,
            ),
        ],
        tooltip={"text": "{address_city}\n{living_space_from} m²\n{bedrooms} slaapkamers"}
    ))


