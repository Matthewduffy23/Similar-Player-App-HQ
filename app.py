# app.py — Player Similarity Finder (simple player select + league presets, 0–100 league bar)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Player Similarity Finder", layout="wide")

# ====== Styling ======
st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
/* Big pill toggle */
div.top-toggle {margin: .25rem 0 1rem 0;}
div.top-toggle .stRadio > div {flex-direction: row; gap: .5rem; flex-wrap: wrap;}
div.top-toggle label {padding: 10px 16px; border-radius: 999px; border:1px solid #E5E7EB;
  background:#F8FAFC; font-weight:700;}
div.top-toggle input:checked + div > label {background:#E5F0FF; border-color:#93C5FD; color:#1D4ED8;}
/* Sidebar cards */
section[data-testid="stSidebar"] {width: 320px !important;}
.sidebar-card {background:#F8FAFC;border:1px solid #E5E7EB;border-radius:12px;padding:.85rem .9rem;margin-bottom:.8rem;}
.sidebar-title {font-weight:700;font-size:0.95rem;margin-bottom:.6rem;display:flex;gap:.4rem;align-items:center;}
/* Preset chips */
.preset-chips .stRadio > div {flex-direction: row; gap: .5rem; flex-wrap: wrap;}
.preset-chips label {padding:8px 10px;border-radius:999px;border:1px solid #E5E7EB;background:#fff;}
.preset-chips input:checked + div > label {background:#111827;color:#fff;border-color:#111827;}
</style>
""", unsafe_allow_html=True)

st.title("⚽ Player Similarity Finder")

# ====== Load data ======
df = pd.read_csv("WORLDJUNE25.csv"



# ====== Load data ======
df = pd.read_csv("WORLDJUNE25.csv")

# ---------------------------
# Position groups
# ---------------------------
def attackers_mask(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    prefixes = ('RWF','LWF','LAMF','RAMF','AMF','RW,','LW,')
    return s.isin(['RW','LW']) | s.str.startswith(prefixes)

def group_mask(series: pd.Series, group: str) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    if group == "Centre Backs": return s.str.startswith(('LCB','RCB','CB'))
    if group == "Full Backs":   return s.str.startswith(('LB','LWB','RB','RWB'))
    if group == "Midfielders":  return s.str.startswith(('LCMF','RCMF','LDMF','RDMF','DMF'))
    if group == "Attackers":    return attackers_mask(series)
    if group == "Strikers":     return s.str.startswith(('CF',))
    return pd.Series([True]*len(series), index=series.index)

# ---------------------------
# Role toggle (top)
# ---------------------------
st.markdown('<div class="top-toggle">', unsafe_allow_html=True)
calc_mode = st.radio(
    "Choose calculation mode",
    ["Centre Backs", "Full Backs", "Midfielders", "Attackers", "Strikers"],
    horizontal=True, label_visibility="collapsed", key="mode_toggle_top"
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Feature sets per role
# ---------------------------
CB_FEATURES = [
    'Successful defensive actions per 90','Defensive duels per 90','Defensive duels won, %',
    'Aerial duels per 90','Aerial duels won, %','Shots blocked per 90','PAdj Interceptions','Dribbles per 90',
    'Successful dribbles, %','Progressive runs per 90','Accelerations per 90','Passes per 90','Accurate passes, %',
    'Forward passes per 90','Accurate forward passes, %','Long passes per 90','Accurate long passes, %',
    'Passes to final third per 90','Accurate passes to final third, %','Progressive passes per 90',
    'Accurate progressive passes, %',
]
FB_FEATURES = [
    'Defensive duels per 90','Defensive duels won, %','Aerial duels per 90','Aerial duels won, %','PAdj Interceptions',
    'Crosses per 90','Accurate crosses, %','Dribbles per 90','Successful dribbles, %','Touches in box per 90',
    'Progressive runs per 90','Accelerations per 90','Passes per 90','Accurate passes, %','Forward passes per 90',
    'Accurate forward passes, %','xA per 90','Smart passes per 90','Key passes per 90',
    'Passes to final third per 90','Accurate passes to final third, %','Passes to penalty area per 90',
    'Deep completions per 90','Progressive passes per 90','Accurate progressive passes, %',
]
CM_FEATURES = [
    'Successful defensive actions per 90','Defensive duels per 90','Defensive duels won, %','Aerial duels per 90',
    'Aerial duels won, %','Shots blocked per 90','PAdj Interceptions','Non-penalty goals per 90','xG per 90',
    'Shots per 90','Shots on target, %','Dribbles per 90','Successful dribbles, %','Offensive duels per 90',
    'Offensive duels won, %','Touches in box per 90','Progressive runs per 90','Accelerations per 90','Passes per 90',
    'Accurate passes, %','Forward passes per 90','Accurate forward passes, %','Long passes per 90',
    'Accurate long passes, %','xA per 90','Smart passes per 90','Key passes per 90','Passes to final third per 90',
    'Accurate passes to final third, %','Passes to penalty area per 90','Accurate passes to penalty area, %',
    'Deep completions per 90','Progressive passes per 90',
]
ATT_FEATURES = [
    'Defensive duels per 90','Aerial duels per 90','Aerial duels won, %','PAdj Interceptions','xG per 90',
    'Non-penalty goals per 90','Shots per 90','Crosses per 90','Accurate crosses, %','Dribbles per 90',
    'Successful dribbles, %','Touches in box per 90','Progressive runs per 90','Accelerations per 90','Passes per 90',
    'Accurate passes, %','xA per 90','Smart passes per 90','Key passes per 90','Passes to final third per 90',
    'Accurate passes to final third, %','Passes to penalty area per 90','Accurate passes to penalty area, %',
    'Deep completions per 90','Progressive passes per 90',
]
ST_FEATURES = [
    'Defensive duels per 90','Aerial duels per 90','Aerial duels won, %','Non-penalty goals per 90','xG per 90',
    'Shots per 90','Shots on target, %','Crosses per 90','Dribbles per 90','Successful dribbles, %',
    'Offensive duels per 90','Touches in box per 90','Progressive runs per 90','Passes per 90','Accurate passes, %',
    'xA per 90','Smart passes per 90','Passes to final third per 90','Passes to penalty area per 90','Deep completions per 90',
]

MODE_FEATURES = {
    "Centre Backs": CB_FEATURES,
    "Full Backs":   FB_FEATURES,
    "Midfielders":  CM_FEATURES,
    "Attackers":    ATT_FEATURES,
    "Strikers":     ST_FEATURES,
}

# ---------------------------
# Default weights per role
# ---------------------------
MODE_WEIGHTS = {
    "Centre Backs": {
        'Passes per 90': 2,'Accurate passes, %': 2,'Progressive passes per 90': 2,
        'Defensive duels per 90': 2,'Defensive duels won, %': 2,'Dribbles per 90': 2,
        'PAdj Interceptions': 2,'Progressive runs per 90': 2,'Aerial duels per 90': 2,
        'Aerial duels won, %': 3,
    },
    "Full Backs": {
        'Passes per 90': 2,'Passes to penalty area per 90': 2,'Dribbles per 90': 2,'xA per 90': 2,
        'Progressive passes per 90': 2,'Defensive duels per 90': 2,'Progressive runs per 90': 2,
        'PAdj Interceptions': 2,'Aerial duels won, %': 2,'Touches in box per 90': 2,
    },
    "Midfielders": {
        'Passes per 90': 2,'Progressive runs per 90': 2,'Progressive passes per 90': 2,'Dribbles per 90': 2,
        'xA per 90': 2,'Touches in box per 90': 2,'Accurate passes, %': 2,'Aerial duels won, %': 2,
        'Passes to penalty area per 90': 2,'Defensive duels per 90': 2,
    },
    "Attackers": {
        'xG per 90': 2,'Shots per 90': 2,'Dribbles per 90': 2,'Crosses per 90': 2,'Non-penalty goals per 90': 2,
        'xA per 90': 2,'Progressive passes per 90': 2,'Defensive duels per 90': 2,'Passes per 90': 2,
        'Passes to penalty area per 90': 2,'Aerial duels won, %': 2,
    },
    "Strikers": {
        'Passes per 90': 3,'Dribbles per 90': 3,'Non-penalty goals per 90': 3,
        'Aerial duels won, %': 2,'Aerial duels per 90': 3,'xA per 90': 2,'xG per 90': 3,'Touches in box per 90': 2,
    },
}

# ---------------------------
# Leagues & presets
# ---------------------------
included_leagues = [
    'England 1.','England 2.','England 3.','England 4.','England 5.','England 6.','England 7.','England 8.','England 9.','England 10.',
    'Albania 1.','Algeria 1.','Andorra 1.','Argentina 1.','Armenia 1.','Australia 1.','Austria 1.','Austria 2.','Azerbaijan 1.',
    'Belgium 1.','Belgium 2.','Bolivia 1.','Bosnia 1.','Brazil 1.','Brazil 2.','Brazil 3.','Bulgaria 1.','Canada 1.','Chile 1.',
    'Colombia 1.','Costa Rica 1.','Croatia 1.','Cyprus 1.','Czech 1.','Czech 2.','Denmark 1.','Denmark 2.','Ecuador 1.',
    'Egypt 1.','Estonia 1.','Finland 1.','France 1.','France 2.','France 3.','Georgia 1.','Germany 1.','Germany 2.','Germany 3.',
    'Germany 4.','Greece 1.','Hungary 1.','Iceland 1.','Israel 1.','Israel 2.','Italy 1.','Italy 2.','Italy 3.','Japan 1.','Japan 2.',
    'Kazakhstan 1.','Korea 1.','Latvia 1.','Lithuania 1.','Malta 1.','Mexico 1.','Moldova 1.','Morocco 1.','Netherlands 1.','Netherlands 2.',
    'North Macedonia 1.','Northern Ireland 1.','Norway 1.','Norway 2.','Paraguay 1.','Peru 1.','Poland 1.','Poland 2.','Portugal 1.',
    'Portugal 2.','Portugal 3.','Qatar 1.','Ireland 1.','Romania 1.','Russia 1.','Saudi 1.','Scotland 1.','Scotland 2.','Scotland 3.',
    'Serbia 1.','Serbia 2.','Slovakia 1.','Slovakia 2.','Slovenia 1.','Slovenia 2.','South Africa 1.','Spain 1.','Spain 2.','Spain 3.',
    'Sweden 1.','Sweden 2.','Switzerland 1.','Switzerland 2.','Tunisia 1.','Turkey 1.','Turkey 2.','Ukraine 1.','UAE 1.','USA 1.','USA 2.',
    'Uruguay 1.','Uzbekistan 1.','Venezuela 1.','Wales 1.'
]

PRESETS = {
    "Top 5 Europe": ['England 1.','France 1.','Germany 1.','Italy 1.','Spain 1.'],
    "Top 20 Europe": [
        'England 1.','Italy 1.','Spain 1.','Germany 1.','France 1.','England 2.','Portugal 1.','Belgium 1.','Turkey 1.',
        'Germany 2.','Spain 2.','France 2.','Netherlands 1.','Austria 1.','Switzerland 1.','Denmark 1.','Croatia 1.','Italy 2.','Czech 1.','Norway 1.'
    ],
    "EFL (England 2–4)": ['England 2.','England 3.','England 4.'],
    "All listed leagues": included_leagues,
    "Custom": None,
}

# 0–100 league strengths
league_strengths = {
    'England 1.': 100.00, 'Italy 1.': 97.14, 'Spain 1.': 94.29, 'Germany 1.': 94.29, 'France 1.': 91.43,
    'Brazil 1.': 82.86, 'England 2.': 71.43, 'Portugal 1.': 71.43, 'Argentina 1.': 71.43, 'Belgium 1.': 68.57,
    'Mexico 1.': 68.57, 'Turkey 1.': 65.71, 'Germany 2.': 65.71, 'Spain 2.': 65.71, 'France 2.': 65.71,
    'USA 1.': 65.71, 'Russia 1.': 65.71, 'Colombia 1.': 62.86, 'Netherlands 1.': 62.86, 'Austria 1.': 62.86,
    'Switzerland 1.': 62.86, 'Denmark 1.': 62.86, 'Croatia 1.': 62.86, 'Japan 1.': 62.86, 'Korea 1.': 62.86,
    'Italy 2.': 62.86, 'Czech 1.': 57.14, 'Norway 1.': 57.14, 'Poland 1.': 57.14, 'Romania 1.': 57.14,
    'Israel 1.': 57.14, 'Algeria 1.': 57.14, 'Paraguay 1.': 57.14, 'Saudi 1.': 57.14, 'Uruguay 1.': 57.14,
    'Morocco 1.': 57.00, 'Brazil 2.': 56.00, 'Ukraine 1.': 54.29, 'Ecuador 1.': 54.29, 'Spain 3.': 54.29,
    'Scotland 1.': 54.29, 'Chile 1.': 51.43, 'Cyprus 1.': 51.43, 'Portugal 2.': 51.43, 'Slovakia 1.': 51.43,
    'Australia 1.': 51.43, 'Hungary 1.': 51.43, 'Egypt 1.': 51.43, 'England 3.': 51.43, 'France 3.': 48.00,
    'Japan 2.': 48.00, 'Bulgaria 1.': 48.57, 'Slovenia 1.': 48.57, 'Venezuela 1.': 48.00, 'Germany 3.': 45.71,
    'Albania 1.': 44.00, 'Serbia 1.': 42.86, 'Belgium 2.': 42.86, 'Bosnia 1.': 42.86, 'Kosovo 1.': 42.86,
    'Nigeria 1.': 42.86, 'Azerbaijan 1.': 50.00, 'Bolivia 1.': 50.00, 'Costa Rica 1.': 50.00, 'South Africa 1.': 50.00, 'UAE 1.': 50.00,
    'Georgia 1.': 40.00, 'Finland 1.': 40.00, 'Italy 3.': 40.00, 'Peru 1.': 40.00, 'Tunisia 1.': 40.00, 'USA 2.': 40.00,
    'Armenia 1.': 40.00, 'North Macedonia 1.': 40.00, 'Qatar 1.': 40.00, 'Uzbekistan 1.': 42.00, 'Norway 2.': 42.00,
    'Kazakhstan 1.': 42.00, 'Poland 2.': 38.00, 'Denmark 2.': 37.00, 'Czech 2.': 37.14, 'Israel 2.': 37.14, 'Netherlands 2.': 37.14, 'Switzerland 2.': 37.14,
    'Iceland 1.': 34.29, 'Ireland 1.': 34.29, 'Sweden 2.': 34.29, 'Germany 4.': 34.29, 'Malta 1.': 30.00, 'Turkey 2.': 35.00, 'Canada 1.': 28.57,
    'England 4.': 28.57, 'Scotland 2.': 28.57, 'Moldova 1.': 28.57, 'Austria 2.': 25.71, 'Lithuania 1.': 25.71, 'Brazil 3.': 25.00, 'England 7.': 25.00,
    'Slovenia 2.': 22.00, 'Latvia 1.': 22.86, 'Serbia 2.': 20.00, 'Slovakia 2.': 20.00, 'England 9.': 20.00, 'England 8.': 15.00,
    'Montenegro 1.': 14.29, 'Wales 1.': 12.00, 'Portugal 3.': 11.43, 'Northern Ireland 1.': 11.43, 'England 5.': 11.43, 'Andorra 1.': 10.00,
    'Estonia 1.': 8.57, 'England 10.': 5.00, 'Scotland 3.': 0.00, 'England 6.': 0.00
}

DEFAULT_PERCENTILE_WEIGHT = 0.7
DEFAULT_LEAGUE_WEIGHT = 0.4

# ---------------------------
# Sidebar (simple)
# ---------------------------
features = MODE_FEATURES[calc_mode]
default_role_weights = MODE_WEIGHTS[calc_mode]

with st.sidebar:
    st.markdown('<div class="sidebar-card"><div class="sidebar-title">🎯 Target player</div>', unsafe_allow_html=True)
    df_role_all = df[group_mask(df['Position'], calc_mode)]
    target_player = st.selectbox("Select player", sorted(df_role_all['Player'].dropna().unique()))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card"><div class="sidebar-title">🧭 Candidate leagues</div>', unsafe_allow_html=True)
    preset_name = st.radio(
        "Preset", ["Top 5 Europe","Top 20 Europe","EFL (England 2–4)","All listed leagues","Custom"],
        index=0, horizontal=True, label_visibility="collapsed", key="preset_radio"
    )
    if preset_name == "Custom":
        leagues_available = sorted(set(included_leagues) | set(df.get('League', pd.Series([])).dropna().unique()))
        leagues_selected = st.multiselect("Select leagues", leagues_available, default=included_leagues)
    else:
        leagues_selected = PRESETS[preset_name]
    st.caption(f"Preset: **{preset_name}** | Selected leagues: **{len(set(leagues_selected))}**")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card"><div class="sidebar-title">⚙️ Filters & weights</div>', unsafe_allow_html=True)
    min_minutes, max_minutes = st.slider("Minutes played", 0, 5000, (500, 5000))
    min_age, max_age = st.slider("Age", 14, 45, (16, 40))
    min_strength, max_strength = st.slider("League quality (strength)", 0, 101, (0, 101))

    percentile_weight = st.slider(
        "Percentile weight",
        0.0, 1.0, DEFAULT_PERCENTILE_WEIGHT, 0.05,
        help="Blend between league-relative percentiles (higher = more context-normalized) and raw values."
    )
    league_weight = st.slider(
        "League weight (difficulty adjustment)",
        0.0, 1.0, DEFAULT_LEAGUE_WEIGHT, 0.05,
        help="Penalizes differences in league strength between the target’s league and candidate leagues."
    )

    with st.expander("Feature weights (prefilled by role)"):
        wf = {}
        for f in features:
            # Default to 1 if not explicitly set for the role
            wf[f] = st.slider(f"{f}", 1, 5, int(default_role_weights.get(f, 1)))
    top_n = st.number_input("Show top N", min_value=5, max_value=200, value=50, step=5)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Compute
# ---------------------------
required_cols = {'Player','Team','League','Age','Position','Minutes played','Market value', *features}
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Your data is missing required columns: {missing}")
    st.stop()

# Candidate pool: by preset leagues + role + feature completeness
df_candidates = df[df['League'].isin(leagues_selected) & group_mask(df['Position'], calc_mode)].copy()
df_candidates = df_candidates.dropna(subset=features)
if df_candidates.empty:
    st.warning("No candidates after filters/preset. Choose a different preset or relax filters.")
    st.stop()

# Target row (from the full dataset but by role only)
df_role_pool = df[group_mask(df['Position'], calc_mode)].copy()
if target_player not in df_role_pool['Player'].values:
    st.warning("Target player not found for this role.")
    st.stop()
target_row = df_role_pool.loc[df_role_pool['Player'] == target_player].iloc[0]
target_league = str(target_row['League'])

target_features = target_row[features].values.reshape(1, -1)

# Percentiles within each league (for candidate pool)
percentile_ranks = df_candidates.groupby('League')[features].rank(pct=True).values
# Target percentiles: compute over the whole dataset by league, then pick target
target_percentiles = (
    df.groupby('League')[features]
      .rank(pct=True)
      .loc[df['Player'] == target_player]
      .values
)

# Feature weights vector
weights = np.array([wf.get(f, 1) for f in features], dtype=float)

# Standardize actual values over candidate pool
scaler = StandardScaler()
standardized_features = scaler.fit_transform(df_candidates[features])
target_features_standardized = scaler.transform(target_features)

# Distances
percentile_distances = np.linalg.norm((percentile_ranks - target_percentiles) * weights, axis=1)
actual_value_distances = np.linalg.norm((standardized_features - target_features_standardized) * weights, axis=1)

combined = percentile_distances * percentile_weight + actual_value_distances * (1.0 - percentile_weight)

# Normalize to similarity 0..100
norm = (combined - np.min(combined)) / (np.ptp(combined) if np.ptp(combined) != 0 else 1.0)
similarities = ((1 - norm) * 100).round(2)

# Build result frame + filters
similarity_df = df_candidates.copy()
similarity_df['Similarity'] = similarities
similarity_df = similarity_df[similarity_df['Player'] != target_player]

# Simple user filters
similarity_df = similarity_df[
    (similarity_df['Minutes played'].between(min_minutes, max_minutes, inclusive='both')) &
    (similarity_df['Age'].between(min_age, max_age, inclusive='both'))
]

# League strength & symmetric penalty (0-100)
similarity_df['League strength'] = similarity_df['League'].map(league_strengths).fillna(0.0)
target_league_strength = float(league_strengths.get(target_league, 1.0))

similarity_df = similarity_df[
    (similarity_df['League strength'] >= float(min_strength)) &
    (similarity_df['League strength'] <= float(max_strength))
]

eps = 1e-6
cand_ls = np.maximum(similarity_df['League strength'].astype(float), eps)
tgt_ls = max(target_league_strength, eps)
league_ratio = np.minimum(cand_ls / tgt_ls, tgt_ls / cand_ls)  # <= 1

similarity_df['Adjusted Similarity'] = (
    similarity_df['Similarity'] * ((1 - league_weight) + league_weight * league_ratio)
)

similarity_df = similarity_df.sort_values('Adjusted Similarity', ascending=False).reset_index(drop=True)
similarity_df.insert(0, 'Rank', np.arange(1, len(similarity_df) + 1))

# ---------------------------
# UI Output
# ---------------------------
st.subheader(f"{calc_mode} — Similar to: {target_player}  |  Target league: {target_league} (strength {target_league_strength:.1f})")

cols_to_show = ['Rank','Player','Team','League','League strength','Age','Minutes played','Adjusted Similarity']
cols_to_show = [c for c in cols_to_show if c in similarity_df.columns]

# League strength as 0–100 progress bar
st.dataframe(
    similarity_df[cols_to_show].head(int(top_n)),
    use_container_width=True,
    column_config={
        "League strength": st.column_config.ProgressColumn(
            "League strength", min_value=0, max_value=100, format="%.1f",
            help="0–100 relative league quality"
        ),
        "Adjusted Similarity": st.column_config.NumberColumn("Adjusted Similarity", format="%.2f"),
        "Minutes played": st.column_config.NumberColumn("Minutes played", format="%d"),
        "Age": st.column_config.NumberColumn("Age", format="%d"),
        "Rank": st.column_config.NumberColumn("Rank", format="%d"),
    }
)

csv = similarity_df[cols_to_show].to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download full results (CSV)", data=csv, file_name="similarity_results.csv", mime="text/csv")

with st.expander("Debug / Repro details"):
    st.write({
        "mode": calc_mode,
        "preset": preset_name,
        "candidate_leagues_final_count": len(set(leagues_selected)),
        "percentile_weight": float(percentile_weight),
        "league_weight": float(league_weight),
        "target_league_strength": float(target_league_strength),
        "n_candidates": int(len(similarity_df)),
        "minutes_range": (int(min_minutes), int(max_minutes)),
        "age_range": (int(min_age), int(max_age)),
        "features_used": features,
        "weights_used": {f: int(wf.get(f, 1)) for f in features},
    })


