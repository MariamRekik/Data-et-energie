"""
app.py — Dashboard Data & Energie:
Dashboard interactif pour le projet de ML sur la consommation energetique.
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

warnings.filterwarnings("ignore")

# Configuration

st.set_page_config(
    page_title="Data & Energie",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "Data" / "RES2-6-9.csv"
LABELS_PATH = BASE_DIR / "Data" / "RES2-6-9-labels.csv"

OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
Results_DIR = OUTPUTS_DIR / "results"



FIG_CLUSTERING_DIR = FIGURES_DIR / "clustering"
FIG_CLASSIFICATION_DIR = FIGURES_DIR / "classification"
FIG_PREDICTION_DIR = FIGURES_DIR / "prediction"
GENERATION_CSV_PATH = Results_DIR/ "courbes_generees_cvae.csv"

COL_PDL = "pdl_id"
COL_DT  = "horodate"
COL_PWR = "puissance_w"

FEATURE_COLS = [
    "active_day_rate", "n_runs", "mean_run_len", "max_run_len",
    "mean_gap_len", "max_gap_len",
    "mean_daily_kwh", "p95_daily_kwh", "cv_daily_kwh",
    "active_rate_weekday", "active_rate_weekend",
    "mean_kwh_weekday", "mean_kwh_weekend",
    "winter_minus_summer", "seasonality_amp",
    "r_global", "r_mid", "r_summer", "r_winter",
]

COLORS = {"RP": "#2196F3", "RS": "#FF5722"}


# CSS

st.markdown("""
<style>

/* ===== Fond général ===== */
.main {
    background-color: #0e1117;
}

/* ===== Texte global ===== */
html, body, [class*="css"] {
    color: #FAFAFA;
}

/* ===== Metric cards ===== */
.metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 4px solid #2196F3;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1a1a2e;
}

.metric-label {
    font-size: 0.9rem;
    color: #444444;
    margin-top: 2px;
}

/* ===== Story box ===== */
.story-box {
    background: linear-gradient(135deg, #e8f4fd, #f0f7ff);
    border-radius: 12px;
    padding: 18px 22px;
    border-left: 5px solid #1565C0;
    margin-bottom: 18px;
    font-size: 1.02rem;
    line-height: 1.7;
    color: #111111;
}

.story-box b {
    color: #0D47A1;
}

/* ===== Why box ===== */
.why-box {
    background: #fff8e1;
    border-radius: 10px;
    padding: 14px 18px;
    border-left: 4px solid #F9A825;
    margin-bottom: 14px;
    color: #111111;
}

.why-box strong {
    color: #E65100;
}

/* ===== Tabs ===== */
.stTabs [data-baseweb="tab"] {
    font-size: 16px;
    font-weight: 600;
}

/* ===== Dataframes ===== */
[data-testid="stDataFrame"] {
    background-color: white;
    color: black;
    border-radius: 10px;
}

/* ===== Infos / warnings ===== */
.stAlert {
    border-radius: 10px;
}

/* ===== Arrow ===== */
.step-arrow {
    text-align: center;
    font-size: 1.8rem;
    color: #90CAF9;
    margin: 0;
    line-height: 1;
}

</style>
""", unsafe_allow_html=True)


# Chargement des donnees
 
@st.cache_data(show_spinner="Chargement des donnees brutes...")
def load_raw_data():
    raw = pd.read_csv(DATA_PATH, sep=None, engine="python")
    raw.columns = [COL_PDL, COL_DT, COL_PWR]
    raw[COL_DT] = pd.to_datetime(raw[COL_DT], utc=True, errors="coerce")
    raw[COL_DT] = raw[COL_DT].dt.tz_convert("Europe/Paris")
    df = raw.dropna(subset=[COL_PDL, COL_DT, COL_PWR]).copy()
    df[COL_PWR] = pd.to_numeric(df[COL_PWR], errors="coerce")
    df = df.dropna(subset=[COL_PWR])
    df["date"]       = df[COL_DT].dt.date
    df["dow"]        = df[COL_DT].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hh_index"]   = df[COL_DT].dt.hour * 2 + df[COL_DT].dt.minute // 30
    df["energy_kwh"] = df[COL_PWR] * 0.5 / 1000
    ref = pd.read_csv(LABELS_PATH, sep=None, engine="python")
    ref.columns = ["pdl_id", "label_rs_rp", "cluster_ref"]
    return df, ref


@st.cache_data(show_spinner="Calcul des features client...")
def compute_features(_df, _ref):
    df, ref = _df.copy(), _ref.copy()

    daily = (
        df.groupby([COL_PDL, "date"])
        .agg(daily_kwh=("energy_kwh", "sum"), is_weekend=("is_weekend", "first"))
        .reset_index()
    )
    daily["date"]  = pd.to_datetime(daily["date"])
    daily["month"] = daily["date"].dt.month

    th = daily.groupby(COL_PDL)["daily_kwh"].quantile(0.2).rename("th_pdl").reset_index()
    daily = daily.merge(th, on=COL_PDL)
    daily["is_active_day"] = (daily["daily_kwh"] > daily["th_pdl"]).astype(int)

    activity = daily.groupby(COL_PDL).agg(
        active_day_rate=("is_active_day", "mean")
    ).reset_index()

    # Runs (boucle explicite — robuste avec pandas 2.x)
    runs_list = []
    for pdl, grp in daily.groupby(COL_PDL)["is_active_day"]:
        vals = grp.values
        runs, gaps, r, g = [], [], 0, 0
        for v in vals:
            if v:
                r += 1
                if g:
                    gaps.append(g); g = 0
            else:
                if r:
                    runs.append(r); r = 0
                g += 1
        if r: runs.append(r)
        if g: gaps.append(g)
        runs_list.append({
            COL_PDL: pdl,
            "n_runs":       float(len(runs)),
            "mean_run_len": float(np.mean(runs) if runs else 0),
            "max_run_len":  float(max(runs)     if runs else 0),
            "mean_gap_len": float(np.mean(gaps) if gaps else 0),
            "max_gap_len":  float(max(gaps)     if gaps else 0),
        })
    runs_stats = pd.DataFrame(runs_list)

    week_list = []
    for pdl, grp in daily.groupby(COL_PDL):
        wd = grp[grp["is_weekend"] == 0]
        we = grp[grp["is_weekend"] == 1]
        week_list.append({
            COL_PDL:               pdl,
            "active_rate_weekday": float(wd["is_active_day"].mean()) if len(wd) else 0.0,
            "active_rate_weekend": float(we["is_active_day"].mean()) if len(we) else 0.0,
            "mean_kwh_weekday":    float(wd["daily_kwh"].mean())     if len(wd) else 0.0,
            "mean_kwh_weekend":    float(we["daily_kwh"].mean())     if len(we) else 0.0,
        })
    week_pattern = pd.DataFrame(week_list)

    season_map = {12:3, 1:3, 2:3, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:1, 10:1, 11:1}
    daily["season"] = daily["month"].map(season_map)
    total = daily.groupby(COL_PDL)["daily_kwh"].sum().rename("total_kwh").reset_index()
    s_agg = (
        daily.groupby([COL_PDL, "season"])["daily_kwh"]
        .sum().unstack(fill_value=0)
        .rename(columns={1: "s_mid", 2: "s_summer", 3: "s_winter"})
        .reset_index()
    )
    for col in ["s_mid", "s_summer", "s_winter"]:
        if col not in s_agg.columns:
            s_agg[col] = 0.0
    s_agg = s_agg.merge(total, on=COL_PDL)
    for col, key in [("s_mid","r_mid"),("s_summer","r_summer"),("s_winter","r_winter")]:
        s_agg[key] = s_agg[col] / s_agg["total_kwh"].clip(lower=1e-9)
    s_agg["r_global"] = 1.0
    season_stats = s_agg[[COL_PDL, "r_global", "r_mid", "r_summer", "r_winter"]]

    daily_stats = (
        daily.groupby(COL_PDL)["daily_kwh"]
        .agg(
            mean_daily_kwh="mean",
            p95_daily_kwh=lambda x: x.quantile(0.95),
            cv_daily_kwh=lambda x: x.std() / x.mean() if x.mean() > 0 else 0,
        )
        .reset_index()
    )

    features_pdl = (
        activity
        .merge(runs_stats,   on=COL_PDL, how="left")
        .merge(week_pattern, on=COL_PDL, how="left")
        .merge(season_stats, on=COL_PDL, how="left")
        .merge(daily_stats,  on=COL_PDL, how="left")
    )
    features_pdl["seasonality_amp"]     = (
        features_pdl[["r_mid","r_summer","r_winter"]].max(axis=1) -
        features_pdl[["r_mid","r_summer","r_winter"]].min(axis=1)
    )
    features_pdl["winter_minus_summer"] = features_pdl["r_winter"] - features_pdl["r_summer"]
    features_pdl = features_pdl.merge(
    ref[[COL_PDL, "label_rs_rp", "cluster_ref"]],
    on=COL_PDL,
    how="inner"
)
    return features_pdl, daily


@st.cache_resource(show_spinner="Entrainement des classifieurs...")
def train_classifiers(_X_train, _y_train):
    models = {
        "Regression Logistique": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "SVM (RBF)": SVC(
            kernel="rbf", class_weight="balanced",
            random_state=RANDOM_STATE, probability=True),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    trained, results = {}, {}
    for name, clf in models.items():
        clf.fit(_X_train, _y_train)
        cv_f1 = cross_val_score(clf, _X_train, _y_train, cv=cv, scoring="f1_macro").mean()
        trained[name] = clf
        results[name] = {"cv_f1_macro": cv_f1}
    return trained, results


@st.cache_data(show_spinner="Application du clustering K-Means...")
def run_clustering(_features_pdl):
    X = (
        _features_pdl[FEATURE_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .values
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=10,
        random_state=RANDOM_STATE,
        n_init=20
    )

    clusters = kmeans.fit_predict(X_scaled)

    clustered = _features_pdl.copy()
    clustered["cluster_kmeans"] = clusters

    silhouette = silhouette_score(X_scaled, clusters)

    ari = adjusted_rand_score(
        clustered["cluster_ref"],
        clustered["cluster_kmeans"]
    )

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)

    clustered["PC1"] = coords[:, 0]
    clustered["PC2"] = coords[:, 1]

    return clustered, silhouette, ari

@st.cache_data(show_spinner="Preparation des donnees de prevision (30 min, RP)...")
def prepare_forecast_data(_df, _ref):
    """
    Prevision multi-output 48 demi-heures sur la serie agregee RP.

    Etapes :
      1. Filtrer les clients RP (label_rs_rp == 0).
      2. Sommer la puissance W de tous les clients RP pour chaque pas 30 min.
      3. Convertir W -> kWh (energie = puissance * 0.5 h / 1000).
      4. Ingenierie de features : calendaire cyclique + lags 1/48/96/336
         + rolling stats 48 (sur shift(1) pour eviter la fuite de donnees).
      5. Construire 48 cibles y_1, ..., y_48 (les 48 demi-heures suivantes).
    """
    HORIZON = 48
    rp_pdls = _ref.loc[_ref["label_rs_rp"] == 0, "pdl_id"].values
    sub = _df[_df[COL_PDL].isin(rp_pdls)].copy()

    # Agregation a la demi-heure : SOMME des clients RP, puis W -> kWh
    ts = (
        sub.groupby(COL_DT)[COL_PWR]
        .sum()
        .rename("kwh_total")
        .reset_index()
    )
    ts["kwh_total"] = ts["kwh_total"] * 0.5 / 1000.0
    ts = ts.sort_values(COL_DT).reset_index(drop=True)

    df_f = ts.copy()

    # Features calendaires (encodage cyclique pour heure et mois)
    df_f["hh_index"]   = df_f[COL_DT].dt.hour * 2 + df_f[COL_DT].dt.minute // 30
    df_f["hh_sin"]     = np.sin(2 * np.pi * df_f["hh_index"] / 48)
    df_f["hh_cos"]     = np.cos(2 * np.pi * df_f["hh_index"] / 48)
    df_f["dow"]        = df_f[COL_DT].dt.dayofweek
    df_f["is_weekend"] = (df_f["dow"] >= 5).astype(int)
    df_f["month"]      = df_f[COL_DT].dt.month
    df_f["month_sin"]  = np.sin(2 * np.pi * df_f["month"] / 12)
    df_f["month_cos"]  = np.cos(2 * np.pi * df_f["month"] / 12)

    # Lags (en pas de 30 min)
    for lag in [1, 48, 96, 336]:
        df_f[f"lag_{lag}"] = df_f["kwh_total"].shift(lag)

    # Rolling statistics sur les 48 derniers pas (24 h), shift(1) anti-fuite
    roll = df_f["kwh_total"].shift(1).rolling(48)
    df_f["roll_mean_48"] = roll.mean()
    df_f["roll_std_48"]  = roll.std()
    df_f["roll_max_48"]  = roll.max()

    # Cibles multi-output : les 48 valeurs suivantes
    for h in range(1, HORIZON + 1):
        df_f[f"y_{h}"] = df_f["kwh_total"].shift(-h)

    feature_cols = [
        "hh_sin", "hh_cos", "dow", "is_weekend",
        "month_sin", "month_cos",
        "lag_1", "lag_48", "lag_96", "lag_336",
        "roll_mean_48", "roll_std_48", "roll_max_48",
    ]
    target_cols = [f"y_{h}" for h in range(1, HORIZON + 1)]

    df_f = df_f.dropna(subset=feature_cols + target_cols).reset_index(drop=True)
    return df_f, ts, feature_cols, target_cols


@st.cache_resource(show_spinner="Entrainement multi-output 48 demi-heures (LR + RF)...")
def train_forecast_models(_df_feat, _feature_cols, _target_cols):
    """
    Entrainement multi-output direct des modeles LR et RF.
      - LinearRegression : supporte nativement 48 sorties (OLS, sans HP).
      - RandomForestRegressor enveloppe par MultiOutputRegressor :
        un RF independant par horizon, en parallele sur les 48 sorties.
        Hyperparametres choisis a partir de la recherche TimeSeriesSplit
        effectuee dans le notebook 03_prediction.ipynb (config raisonnable
        pour le dashboard ; le notebook cherche la valeur optimale).
      - LSTM : omis dans le dashboard pour le temps d'execution
        (resultats complets disponibles dans 03_prediction.ipynb).
    Metriques MAE/RMSE/MAPE calculees sur l'ensemble des points (Y aplati).
    """
    HORIZON = 48
    # Split chronologique 80/20, aligne sur des jours complets (multiples de 48).
    split_raw = int(len(_df_feat) * 0.8)
    split     = (split_raw // HORIZON) * HORIZON
    train     = _df_feat.iloc[:split]
    test      = _df_feat.iloc[split:]

    X_tr = train[_feature_cols].values
    Y_tr = train[_target_cols].values
    X_te = test[_feature_cols].values
    Y_te = test[_target_cols].values
    test_dates = test[COL_DT].values

    # Regression Lineaire (baseline multi-output natif)
    lr = LinearRegression().fit(X_tr, Y_tr)

    # Random Forest enveloppe : 48 estimateurs RF entraines en parallele
    rf = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators     = 60,
            max_depth        = 12,
            min_samples_leaf = 2,
            random_state     = RANDOM_STATE,
            n_jobs           = 1,
        ),
        n_jobs=-1,
    ).fit(X_tr, Y_tr)

    results = {}
    for name, model in [("Regression Lineaire", lr), ("Random Forest", rf)]:
        Y_pred = model.predict(X_te)
        flat_t = Y_te.flatten()
        flat_p = Y_pred.flatten()
        mae  = mean_absolute_error(flat_t, flat_p)
        rmse = np.sqrt(mean_squared_error(flat_t, flat_p))
        mask = flat_t > 0
        mape = float(np.mean(np.abs((flat_t[mask] - flat_p[mask]) / flat_t[mask])) * 100)
        results[name] = {
            "mae": float(mae), "rmse": float(rmse), "mape": mape,
            "Y_pred": Y_pred, "Y_true": Y_te,
            "test_dates": test_dates,
        }
    return results

@st.cache_data(show_spinner="Chargement des courbes générées CVAE...")
def load_generated_curves():
    if not GENERATION_CSV_PATH.exists():
        return None

    gen = pd.read_csv(GENERATION_CSV_PATH)

    required_cols = {
        "sample_id", "label", "season",
        "hh_index", "time_h",
        "puissance_w", "energy_kwh_step"
    }

    if not required_cols.issubset(set(gen.columns)):
        st.error(
            "Le fichier courbes_generees_cvae.csv ne contient pas les colonnes attendues."
        )
        return None

    gen["sample_id"] = gen["sample_id"].astype(str)
    gen["label"] = gen["label"].astype(str)
    gen["season"] = gen["season"].astype(str)

    gen["hh_index"] = pd.to_numeric(gen["hh_index"], errors="coerce")
    gen["time_h"] = pd.to_numeric(gen["time_h"], errors="coerce")
    gen["puissance_w"] = pd.to_numeric(gen["puissance_w"], errors="coerce")
    gen["energy_kwh_step"] = pd.to_numeric(gen["energy_kwh_step"], errors="coerce")

    gen = gen.dropna(
        subset=["hh_index", "time_h", "puissance_w", "energy_kwh_step"]
    )

    return gen

# Sidebar
 
with st.sidebar:
    st.markdown("## ⚡ Data & Energie")
    st.markdown("* Ecole des Ponts*")
    st.divider()
    page = st.radio(
        "Navigation",
        ["🏠 Accueil",
         "🔍 Les donnees",
          "🧩 Clustering",
         "🤖 Qui est le client ?",
         "📈 Que va-t-il consommer ?",
         "🎨 Generer de nouveaux profils"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("500 clients · ~8,7 M mesures · pas 30 min")


# Verification des fichiers 
data_ok = os.path.exists(DATA_PATH) and os.path.exists(LABELS_PATH)
#if not data_ok:
 #   st.error(
   #     f"Fichiers de donnees introuvables. "
   #     f"Placez `{DATA_PATH}` et `{LABELS_PATH}` dans le meme dossier que `app.py`."
  #  )
  #  st.stop()

df, ref = load_raw_data()
features_pdl, daily = compute_features(df, ref)



# PAGE 1 : ACCUEIL

if page == "🏠 Accueil":
    st.title("⚡ Comprendre et anticiper la consommation electrique")
    st.markdown(
        "**Un gestionnaire de reseau electrique doit equilibrer en permanence la production "
        "et la consommation.** Si la demande est mal anticipee, cela provoque des surcharges, "
        "des coupures, ou du gaspillage energetique. Ce projet montre comment le Machine "
        "Learning, applique aux donnees de compteurs Linky, peut aider a resoudre ce probleme."
    )
    st.divider()

    # Metriques cles
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-value">500</div>'
            '<div class="metric-label">Clients residentiels suivis</div>'
            '</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-value">428</div>'
            '<div class="metric-label">Clients RP — profil standard</div>'
            '</div>', unsafe_allow_html=True)
    with c3:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-value">72</div>'
            '<div class="metric-label">Clients RS — Residence Secondaire</div>'
            '</div>', unsafe_allow_html=True)
    with c4:
        n_rows = f"{len(df)/1e6:.1f} M"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{n_rows}</div>'
            f'<div class="metric-label">Mesures enregistrees</div>'
            f'</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### Notre approche en 3 etapes")

    col_a, arrow1, col_b, arrow2, col_c = st.columns([5, 1, 5, 1, 5])

    with col_a:
        st.info(
            "#### 🤖 Etape 1 — Identifier le client\n\n"
            "Avant de pouvoir predire la consommation d'un client, il faut "
            "savoir **qui il est**. Une résidence principale (RP) est occupée toute "
            "l'année, tandis qu'une résidence secondaire (RS) n'est habitée qu'en "
            "week-ends ou en vacances — deux profils très distincts. "
            "automatiquement ce type."
        )
    with arrow1:
        st.markdown('<div class="step-arrow" style="margin-top:80px">→</div>',
                    unsafe_allow_html=True)
    with col_b:
        st.info(
            "#### 📈 Etape 2 — Predire sa consommation\n\n"
            "Connaissant le type de client, on peut anticiper **combien "
            "d'electricite il va consommer demain**. Le gestionnaire de "
            "reseau peut ainsi ajuster la production en avance, eviter les "
            "pics de charge et optimiser les couts."
        )
    with arrow2:
        st.markdown('<div class="step-arrow" style="margin-top:80px">→</div>',
                    unsafe_allow_html=True)
    with col_c:
        st.info(
            "#### 🎨 Etape 3 — Simuler des scenarios\n\n"
            "Un modele generatif (CVAE) apprend la forme des courbes de "
            "consommation et peut en **creer de nouvelles**, realistes et "
            "conditionnees au type de client. Cela permet de tester des "
            "scenarios futurs sans avoir besoin de nouvelles donnees reelles."
        )

    st.divider()
    st.markdown(
        '<div class="why-box">'
        '<strong>Pourquoi ce projet est utile pour le reseau electrique ?</strong><br>'
        'En France, Enedis gere 35 millions de compteurs Linky. Identifier rapidement '
        'le profil d\'un client, predire sa consommation et simuler l\'impact de '
        'nouveaux usages (vehicules electriques, mobilite...) est essentiel '
        'pour planifier les investissements et garantir la stabilite du reseau.'
        '</div>', unsafe_allow_html=True)

    st.markdown("**Naviguez dans le menu de gauche pour explorer chaque etape.**")



# PAGE 2 : LES DONNEES

elif page == "🔍 Les donnees":
    st.title("🔍 Les donnees : qui consomme quoi ?")

    st.markdown(
        '<div class="story-box">'
        'Nous disposons de <b>8,7 millions de mesures de puissance</b> enregistrees toutes les '
        '30 minutes pendant environ un an, pour 500 clients residentiels en France. '
        'Ces donnees proviennent de compteurs intelligents Linky. '
        'La premiere chose que l\'on observe : <b>deux comportements tres distincts</b> selon '
        'si la résidence est principale (habitée toute l\'année) ou secondaire (vacances/week-ends).'
        '</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Vue globale", "👤 Profil d'un client", "🏠 Occupation et consommation"])

    #  Tab 1 : Vue globale 
    with tab1:
        st.subheader("Consommation journaliere moyenne — tous clients")
        st.caption(
            "On observe clairement la saisonnalite : forte consommation en hiver "
            "(chauffage), basse en ete. Ce patron est central pour la prevision."
        )
        agg_all = daily.groupby("date")["daily_kwh"].mean().reset_index()
        agg_all["date"] = pd.to_datetime(agg_all["date"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=agg_all["date"], y=agg_all["daily_kwh"],
            mode="lines", name="Conso moyenne",
            line=dict(color="#2196F3", width=1.5),
            fill="tozeroy", fillcolor="rgba(33,150,243,0.08)"
        ))
        fig.update_layout(xaxis_title="Date", yaxis_title="kWh/jour",
                          hovermode="x unified", height=320,
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        col_pie, col_stats = st.columns([1, 2])
        with col_pie:
            fig3 = go.Figure(go.Pie(
                labels=["RP (Principale)", "RS (Secondaire)"],
                values=[428, 72],
                marker_colors=[COLORS["RP"], COLORS["RS"]],
                hole=0.45,
            ))
            fig3.update_layout(height=260, margin=dict(l=0,r=0,t=30,b=0),
                               title="Repartition des 500 clients")
            st.plotly_chart(fig3, use_container_width=True)

        with col_stats:
            st.markdown("#### Differences moyennes entre RS et RP")
            feat_ref = features_pdl.copy()
            feat_ref["Type"] = feat_ref["label_rs_rp"].map({0: "RP", 1: "RS"})
            stats = (
                feat_ref.groupby("Type")[["mean_daily_kwh","p95_daily_kwh","cv_daily_kwh"]]
                .mean().round(3)
                .rename(columns={
                    "mean_daily_kwh": "Conso moy. (kWh/j)",
                    "p95_daily_kwh":  "Conso P95 (kWh/j)",
                    "cv_daily_kwh":   "Variabilite",
                })
            )
            st.dataframe(stats, use_container_width=True)
            st.caption(
                "Les clients RS (Résidences Secondaires) ont une consommation journaliere "
                "plus faible en moyenne car la maison est souvent vide en semaine, "
                "mais avec des pics marques pendant les vacances et les week-ends."
            )

    #  Tab 2 : Profil client 
    with tab2:
        st.subheader("Courbe de consommation individuelle")
        st.caption("Selectionnez un client pour voir sa courbe sur l'annee entiere.")
        all_pdls = sorted(df[COL_PDL].unique())
        pdl_sel  = st.selectbox("Client (identifiant PDL)", all_pdls)
        client_type = ref.loc[ref["pdl_id"] == pdl_sel, "label_rs_rp"].values
        if len(client_type):
            badge = "🔵 Residence Principale (RP)" if client_type[0] == 0 \
                    else "🔴 Residence Secondaire (RS)"
            st.markdown(f"**Type detecte : {badge}**")

        client_daily = daily[daily[COL_PDL] == pdl_sel].copy()
        client_daily["date"] = pd.to_datetime(client_daily["date"])
        color = COLORS["RP"] if (len(client_type) and client_type[0] == 0) else COLORS["RS"]
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(
            x=client_daily["date"], y=client_daily["daily_kwh"],
            mode="lines", name="Consommation",
            line=dict(color=color, width=1.5),
            fill="tozeroy", fillcolor="rgba(33,150,243,0.08)" if color == COLORS["RP"] else "rgba(255,87,34,0.08)"
        ))
        fig_c.update_layout(xaxis_title="Date", yaxis_title="kWh/jour",
                            height=340, hovermode="x unified",
                            margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_c, use_container_width=True)

    #  Tab 3 : Occupation et consommation 
    with tab3:
        st.subheader("Le profil intra-journalier revele le mode d'occupation")
        st.markdown(
            '<div class="why-box">'
            '<strong>Observation cle :</strong> les clients RP (Résidences Principales) '
            'presentent une consommation <b>plus reguliere sur toute la semaine</b>. '
            'Leurs occupants sont présents a domicile en semaine et en week-end. '
            'Les RS (Résidences Secondaires) ont des pics concentres sur les week-ends '
            'et les vacances — signal tres discriminant pour les classifier.'
            '</div>', unsafe_allow_html=True)

        intraday = (
            df.merge(ref[["pdl_id","label_rs_rp"]], on="pdl_id", how="left")
            .groupby(["label_rs_rp","hh_index"])["energy_kwh"]
            .mean().reset_index()
        )
        intraday["Type"]  = intraday["label_rs_rp"].map({0: "RP", 1: "RS"})
        intraday["Heure"] = intraday["hh_index"].apply(
            lambda x: f"{x//2:02d}:{(x%2)*30:02d}")

        fig_id = go.Figure()
        for label, color in COLORS.items():
            sub = intraday[intraday["Type"] == label]
            fig_id.add_trace(go.Scatter(
                x=sub["Heure"], y=sub["energy_kwh"] * 1000,
                mode="lines+markers", name=label,
                line=dict(color=color, width=2.5),
                marker=dict(size=4),
            ))
        fig_id.add_vrect(x0="10:00", x1="16:00",
                         fillcolor="rgba(255,235,59,0.15)", line_width=0,
                         annotation_text="Heure de bureau",
                         annotation_position="top left")
        fig_id.update_layout(
            xaxis_title="Heure de la journee",
            yaxis_title="Energie moyenne (Wh / 30 min)",
            height=400, hovermode="x unified",
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig_id, use_container_width=True)

        st.info(
            "**Ce que cela signifie pour le reseau :** un client RP injecte de l'electricite "
            "sur le reseau en milieu de journee. Sans detection automatique de ce type de "
            "client, le gestionnaire de reseau risque de mal anticiper les flux d'energie "
            "locaux, ce qui peut creer des instabilites."
        )


# PAGE 3 : CLUSTERING

elif page == "🧩 Clustering":
    st.title("🧩 Clustering — Découvrir automatiquement des profils")

    st.markdown(
        '<div class="story-box">'
        '<b>Objectif :</b> regrouper les clients selon leur comportement de consommation, '
        'sans utiliser directement les labels RP/RS. '
        'Le clustering permet d’identifier des profils naturels : clients réguliers, '
        'clients intermittents, clients très saisonniers, etc.'
        '</div>',
        unsafe_allow_html=True
    )

    clustered, silhouette, ari = run_clustering(features_pdl)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Nombre de clusters", "10")

    with c2:
        st.metric("Silhouette score", f"{silhouette:.3f}")

    with c3:
        st.metric("ARI vs référence prof", f"{ari:.3f}")

    st.markdown(
        '<div class="why-box">'
        '<strong>Interprétation :</strong> K-Means ne connaît pas les labels RP/RS. '
        'Il crée uniquement des groupes de comportement. Ensuite, on compare ces groupes '
        'avec le fichier de référence du professeur pour voir s’ils correspondent plutôt '
        'à des résidences principales ou secondaires.'
        '</div>',
        unsafe_allow_html=True
    )

    tab_pca, tab_dist, tab_cross, tab_features, tab_figs = st.tabs(
        [
            "🗺️ PCA interactive",
            "📊 Taille des clusters",
            "🏠 Clusters vs RP/RS",
            "📋 Profil moyen",
            "🖼️ Figures notebook"
        ]
    )

    with tab_pca:
        st.subheader("Projection PCA des clients colorés par cluster")

        fig_pca_cluster = px.scatter(
            clustered,
            x="PC1",
            y="PC2",
            color=clustered["cluster_kmeans"].astype(str),
            hover_data=[COL_PDL, "label_rs_rp", "cluster_ref"],
            title="Projection PCA des clusters K-Means",
            labels={
                "color": "Cluster K-Means",
                "PC1": "Composante principale 1",
                "PC2": "Composante principale 2",
            },
        )
        fig_pca_cluster.update_layout(height=500)
        st.plotly_chart(fig_pca_cluster, use_container_width=True)

        st.caption(
            "La PCA permet de projeter les 19 features comportementales en deux dimensions. "
            "Elle sert à visualiser si les clusters sont séparables."
        )

    with tab_dist:
        st.subheader("Nombre de clients par cluster")

        cluster_counts = (
            clustered["cluster_kmeans"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        cluster_counts.columns = ["cluster_kmeans", "nombre_clients"]

        fig_counts = px.bar(
            cluster_counts,
            x="cluster_kmeans",
            y="nombre_clients",
            title="Répartition des clients par cluster",
            labels={
                "cluster_kmeans": "Cluster",
                "nombre_clients": "Nombre de clients",
            },
        )

        st.plotly_chart(fig_counts, use_container_width=True)
        st.dataframe(cluster_counts, use_container_width=True, hide_index=True)

    with tab_cross:
        st.subheader("Comparaison des clusters avec les labels RP/RS")

        clustered["Type"] = clustered["label_rs_rp"].map({
            0: "RP",
            1: "RS"
        })

        cross = pd.crosstab(
            clustered["cluster_kmeans"],
            clustered["Type"],
            normalize="index"
        ).round(3)

        st.dataframe(cross, use_container_width=True)

        fig_cross = px.imshow(
            cross,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Composition RP/RS de chaque cluster"
        )
        fig_cross.update_layout(height=450)
        st.plotly_chart(fig_cross, use_container_width=True)

        st.info(
            "Un cluster avec une forte proportion de RS correspond probablement à un profil "
            "intermittent ou saisonnier. Un cluster dominé par les RP correspond plutôt à une "
            "consommation régulière."
        )

    with tab_features:
        st.subheader("Profil moyen des clusters")

        cluster_profile = (
            clustered
            .groupby("cluster_kmeans")[FEATURE_COLS]
            .mean()
            .round(3)
            .reset_index()
        )

        st.dataframe(cluster_profile, use_container_width=True)

        selected_feature = st.selectbox(
            "Choisir une feature à comparer entre clusters",
            FEATURE_COLS
        )

        fig_feature = px.bar(
            cluster_profile,
            x="cluster_kmeans",
            y=selected_feature,
            title=f"Comparaison de {selected_feature} par cluster",
            labels={
                "cluster_kmeans": "Cluster",
                selected_feature: selected_feature,
            },
        )

        st.plotly_chart(fig_feature, use_container_width=True)

    with tab_figs:
        st.subheader("Figures générées dans le notebook de clustering")

        clustering_figs = {
            "Silhouette score": FIG_CLUSTERING_DIR / "fig_01_silhouette.png",
            "Projection PCA": FIG_CLUSTERING_DIR / "fig_02_pca.png",
            "Profils intra-journaliers": FIG_CLUSTERING_DIR / "fig_03_intraday_profiles.png",
        }

        existing_clustering_figs = {
            name: path for name, path in clustering_figs.items() if path.exists()
        }

        if existing_clustering_figs:
            selected_fig = st.selectbox(
                "Sélectionner une figure de clustering",
                list(existing_clustering_figs.keys())
            )
            st.image(existing_clustering_figs[selected_fig], use_container_width=True)
        else:
            st.warning("Aucune figure de clustering trouvée dans outputs/figures/clustering.")

# PAGE 4 : CLASSIFICATION

elif page == "🤖 Qui est le client ?":
    st.title("🤖 Etape 1 — Identifier automatiquement le type de client")

    st.markdown(
        '<div class="story-box">'
        '<b>Problème :</b> parmi 500 clients, 72 sont des résidences secondaires (RS). '
        'Leur comportement est différent des résidences principales (RP) '
        'et doit être traité séparément pour la prévision et la gestion du réseau. '
        '<br><br>'
        '<b>Solution :</b> nous calculons 19 caractéristiques à partir de '
        'l’historique annuel de chaque client — taux d’activité, saisonnalité, '
        'patterns semaine/week-end — puis nous entraînons plusieurs modèles supervisés '
        'pour prédire RP ou RS.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="why-box">'
        '<strong>Point méthodologique :</strong> contrairement au clustering, la classification '
        'utilise les labels RP/RS comme cible supervisée. Le modèle apprend donc à reproduire '
        'ces labels à partir des features comportementales.'
        '</div>',
        unsafe_allow_html=True
    )

    X = features_pdl[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y = features_pdl["label_rs_rp"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    trained_models, cv_results = train_classifiers(X_train_sc, y_train)

    rows = []
    for name, clf in trained_models.items():
        y_pred = clf.predict(X_test_sc)
        rep = classification_report(
            y_test,
            y_pred,
            target_names=["RP", "RS"],
            output_dict=True
        )

        rows.append({
            "Modèle": name,
            "F1-RS": round(rep["RS"]["f1-score"], 3),
            "F1-macro": round(rep["macro avg"]["f1-score"], 3),
            "Précision-RS": round(rep["RS"]["precision"], 3),
            "Rappel-RS": round(rep["RS"]["recall"], 3),
            "CV F1-macro": round(cv_results[name]["cv_f1_macro"], 3),
        })

    results_df = pd.DataFrame(rows).sort_values("F1-RS", ascending=False)
    best_model = results_df.iloc[0]["Modèle"]
    best_f1 = results_df.iloc[0]["F1-RS"]

    st.success(
        f"Meilleur modèle : **{best_model}** — F1-RS = **{best_f1:.3f}**. "
        f"Cette métrique est importante car les résidences secondaires sont minoritaires."
    )

    tab_scores, tab_cm, tab_fi, tab_pca, tab_figs = st.tabs(
        [
            "📊 Scores modèles",
            "🔲 Matrices de confusion",
            "🌲 Variables importantes",
            "🗺️ PCA RP/RS",
            "🖼️ Figures notebook"
        ]
    )

    with tab_scores:
        st.subheader("Comparaison des modèles")

        st.caption(
            "Le F1-RS mesure la capacité du modèle à détecter les résidences secondaires. "
            "C’est plus pertinent que l’accuracy seule car les RS représentent une classe minoritaire."
        )

        st.dataframe(results_df, use_container_width=True, hide_index=True)

        fig_bar = go.Figure()

        for metric, color in [
            ("F1-RS", "#2196F3"),
            ("F1-macro", "#4CAF50"),
            ("CV F1-macro", "#FF9800")
        ]:
            fig_bar.add_trace(go.Bar(
                x=results_df["Modèle"],
                y=results_df[metric],
                name=metric,
                marker_color=color,
                opacity=0.85
            ))

        fig_bar.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 1.05]),
            height=360,
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", y=1.08)
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_cm:
        st.subheader("Matrices de confusion")

        st.caption(
            "La matrice de confusion montre les bonnes et mauvaises classifications. "
            "Elle permet notamment de vérifier si les RS sont bien retrouvées."
        )

        cols_cm = st.columns(2)

        for idx, (name, clf) in enumerate(trained_models.items()):
            y_pred = clf.predict(X_test_sc)
            cm = confusion_matrix(y_test, y_pred)

            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=["Prédit RP", "Prédit RS"],
                y=["Réel RP", "Réel RS"],
                color_continuous_scale="Blues",
                title=name
            )

            fig_cm.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
            fig_cm.update_coloraxes(showscale=False)

            with cols_cm[idx % 2]:
                st.plotly_chart(fig_cm, use_container_width=True)

    with tab_fi:
        st.subheader("Variables les plus discriminantes")

        rf_clf = trained_models.get("Random Forest")

        if rf_clf:
            imp = pd.DataFrame({
                "Feature": FEATURE_COLS,
                "Importance": rf_clf.feature_importances_,
            }).sort_values("Importance", ascending=True)

            labels_fr = {
                "active_day_rate": "Taux de jours actifs",
                "n_runs": "Nb séquences d'activité",
                "mean_run_len": "Durée moy. séquence",
                "max_run_len": "Durée max. séquence",
                "mean_gap_len": "Durée moy. interruption",
                "max_gap_len": "Durée max. interruption",
                "mean_daily_kwh": "Conso journalière moy.",
                "p95_daily_kwh": "Conso P95",
                "cv_daily_kwh": "Variabilité",
                "active_rate_weekday": "Taux actif semaine",
                "active_rate_weekend": "Taux actif week-end",
                "mean_kwh_weekday": "Conso moy. semaine",
                "mean_kwh_weekend": "Conso moy. week-end",
                "winter_minus_summer": "Écart hiver-été",
                "seasonality_amp": "Amplitude saisonnière",
                "r_global": "Part globale",
                "r_mid": "Part saison intermédiaire",
                "r_summer": "Part été",
                "r_winter": "Part hiver",
            }

            imp["Label"] = imp["Feature"].map(labels_fr).fillna(imp["Feature"])

            fig_fi = go.Figure(go.Bar(
                x=imp["Importance"],
                y=imp["Label"],
                orientation="h",
                marker_color="#2196F3",
                text=imp["Importance"].apply(lambda v: f"{v:.3f}"),
                textposition="outside"
            ))

            fig_fi.update_layout(
                xaxis_title="Importance — Random Forest",
                height=560,
                margin=dict(l=0, r=0, t=20, b=0)
            )

            st.plotly_chart(fig_fi, use_container_width=True)

            st.info(
                "Même si le meilleur modèle peut être un SVM, le Random Forest est utile "
                "pour interpréter les variables importantes. Les durées d’absence, les séquences "
                "d’activité et la saisonnalité sont souvent très discriminantes."
            )

    with tab_pca:
        st.subheader("Projection PCA des clients RP/RS")

        X_sc_all = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        coords = pca.fit_transform(X_sc_all)

        pca_df = pd.DataFrame({
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "Type": ["RP" if v == 0 else "RS" for v in y],
        })

        fig_pca = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="Type",
            color_discrete_map=COLORS,
            opacity=0.7,
            labels={
                "PC1": f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)",
                "PC2": f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)",
            },
            title="Projection PCA des profils RP/RS"
        )

        fig_pca.update_traces(marker=dict(size=7))
        fig_pca.update_layout(height=460, margin=dict(l=0, r=0, t=40, b=0))

        st.plotly_chart(fig_pca, use_container_width=True)

    with tab_figs:
        st.subheader("Figures générées dans le notebook de classification")

        classification_figs = {
            "Importance des variables": FIG_CLASSIFICATION_DIR / "fig_05_feature_importance.png",
            "PCA RP/RS": FIG_CLASSIFICATION_DIR / "fig_06_pca_rs_rp.png",
            "Grille SVM": FIG_CLASSIFICATION_DIR / "fig_svm_grid.png",
        }

        existing_classification_figs = {
            name: path for name, path in classification_figs.items() if path.exists()
        }

        if existing_classification_figs:
            selected_fig = st.selectbox(
                "Sélectionner une figure de classification",
                list(existing_classification_figs.keys())
            )
            st.image(existing_classification_figs[selected_fig], use_container_width=True)
        else:
            st.warning("Aucune figure de classification trouvée dans outputs/figures/classification.")

# PAGE 5 : PREVISION
elif page == "📈 Que va-t-il consommer ?":
    st.title("📈 Étape 2 — Prédire la consommation future d'un client")

    st.markdown(
        '''<div class="story-box">
        <b>Objectif :</b> à partir des <b>14 derniers jours</b> d'un client,
        prédire sa consommation pour les <b>2 jours suivants</b> (96 demi-heures).<br><br>
        Quatre approches sont comparées : deux <b>baselines</b> (récente et hebdomadaire)
        et deux <b>modèles résiduels</b> (Ridge et CNN 1D). Les résultats sont évalués
        par segment <b>RP / RS</b> via le <b>WMAPE</b>.
        </div>''',
        unsafe_allow_html=True
    )
    st.markdown(
        '''<div class="why-box">
        <strong>Approche résiduelle :</strong> les modèles n'apprennent pas directement
        la courbe future — ils apprennent la <em>correction</em> à apporter à une baseline.
        Cela stabilise l'apprentissage et tire profit des forts patterns hebdomadaires.
        </div>''',
        unsafe_allow_html=True
    )

    #  Chargement des résultats pré-calculés 
    RESULTS_FILE = Results_DIR / "prediction_results.csv"
    WINDOWS_FILE = Results_DIR / "prediction_windows_sample.csv"

    tab_results, tab_client, tab_figs, tab_limits = st.tabs([
        "📊 Résultats par segment",
        "👤 Prédiction par client",
        "🖼️ Figures notebook",
        "⚠️ Limites",
    ])

    #  Tab 1 : Résultats 
    with tab_results:
        st.subheader("Comparaison des modèles — WMAPE par segment")
        st.caption(
            "Résultats calculés sur le jeu de test (30 derniers jours). "
            "Le WMAPE mesure l'erreur relative pondérée : il est robuste aux créneaux à consommation nulle."
        )

        if RESULTS_FILE.exists():
            test_results = pd.read_csv(RESULTS_FILE)

            # Tableau principal
            display_cols = ["modele", "segment", "n_examples", "MAE", "RMSE", "WMAPE", "sMAPE"]
            display_cols = [c for c in display_cols if c in test_results.columns]
            st.dataframe(
                test_results[display_cols].sort_values(["segment", "WMAPE"]),
                use_container_width=True,
                hide_index=True,
            )

            # Graphique barres groupées par segment
            for seg in ["RS", "RP", "ALL"]:
                df_seg = test_results[test_results["segment"] == seg].sort_values("WMAPE")
                if df_seg.empty:
                    continue

                fig_bar = go.Figure()
                colors_bar = {
                    "baseline_recent" : "#9E9E9E",
                    "baseline_weekly" : "#FF9800",
                    "ridge_residual"  : "#2196F3",
                    "lgbm_residual"   : "#4CAF50",
                    "cnn_residual"    : "#E91E63",
                }
                for _, row in df_seg.iterrows():
                    fig_bar.add_trace(go.Bar(
                        x=[row["modele"]],
                        y=[row["WMAPE"]],
                        name=row["modele"],
                        marker_color=colors_bar.get(row["modele"], "#607D8B"),
                        text=f"{row['WMAPE']:.1f}%",
                        textposition="outside",
                    ))
                fig_bar.update_layout(
                    title=f"WMAPE — segment {seg}",
                    yaxis_title="WMAPE (%)",
                    showlegend=False,
                    height=300,
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.info(
                "Les résultats ne sont pas encore disponibles. "
                "Lancez le notebook `03_prediction.ipynb` pour les générer, puis "
                f"sauvegardez le DataFrame `test_results` dans : `{RESULTS_FILE}`"
            )
            st.code(
                "# Ajoutez à la fin du notebook :\n"
                "test_results.to_csv(Results_DIR / 'prediction_results.csv', index=False)",
                language="python"
            )

    #  Tab 2 : Prédiction par client 
    with tab_client:
        st.subheader("Visualisation de la prédiction pour un client")

        if WINDOWS_FILE.exists():
            windows_sample = pd.read_csv(WINDOWS_FILE)


            # Filtres
            col_seg, col_pdl = st.columns([1, 3])
            with col_seg:
                seg_filter = st.selectbox("Segment", ["Tous", "RP", "RS"])
            with col_pdl:
                pdl_list = sorted(windows_sample["pdl_id"].unique())
                if seg_filter != "Tous":
                    pdl_list = sorted(
                        windows_sample.loc[windows_sample["segment"] == seg_filter, "pdl_id"].unique()
                    )
                pdl_sel = st.selectbox("Client (PDL)", pdl_list)

            # Filtrage
            sub = windows_sample[windows_sample["pdl_id"] == pdl_sel].copy()
            if "forecast_start" in sub.columns:
                sub["forecast_start"] = pd.to_datetime(sub["forecast_start"])
                dates_avail = sub["forecast_start"].dt.strftime("%d/%m/%Y %Hh").tolist()
                chosen_date = st.select_slider(
                    "Fenêtre de prédiction",
                    options=dates_avail,
                    value=dates_avail[len(dates_avail) // 2] if dates_avail else dates_avail[0]
                )
                row = sub[sub["forecast_start"].dt.strftime("%d/%m/%Y %Hh") == chosen_date].iloc[0]
            else:
                row = sub.iloc[len(sub) // 2]

            # Reconstruction des séries depuis les colonnes
            HORIZON_PLOT = 96
            hh_axis = [f"{h//2:02d}:{(h%2)*30:02d}" for h in range(HORIZON_PLOT)]

            y_cols       = [c for c in sub.columns if c.startswith("y_")]
            recent_cols  = [c for c in sub.columns if c.startswith("b_recent_")]
            weekly_cols  = [c for c in sub.columns if c.startswith("b_weekly_")]
            ridge_cols   = [c for c in sub.columns if c.startswith("ridge_")]
            lgbm_cols    = [c for c in sub.columns if c.startswith("lgbm_")]
            cnn_cols     = [c for c in sub.columns if c.startswith("cnn_")]

            fig_client = go.Figure()

            if y_cols:
                y_vals = row[sorted(y_cols, key=lambda c: int(c.split("_")[-1]))].values.astype(float)
                fig_client.add_trace(go.Scatter(
                    x=hh_axis[:len(y_vals)], y=y_vals,
                    mode="lines+markers", name="Réel",
                    line=dict(color="#1a1a2e", width=2.5), marker=dict(size=3)
                ))
            if recent_cols:
                v = row[sorted(recent_cols, key=lambda c: int(c.split("_")[-1]))].values.astype(float)
                fig_client.add_trace(go.Scatter(x=hh_axis[:len(v)], y=v, mode="lines",
                    name="Baseline récente", line=dict(color="#9E9E9E", dash="dot", width=1.8)))
            if weekly_cols:
                v = row[sorted(weekly_cols, key=lambda c: int(c.split("_")[-1]))].values.astype(float)
                fig_client.add_trace(go.Scatter(x=hh_axis[:len(v)], y=v, mode="lines",
                    name="Baseline hebdo", line=dict(color="#FF9800", dash="dot", width=1.8)))
            if ridge_cols:
                v = row[sorted(ridge_cols, key=lambda c: int(c.split("_")[-1]))].values.astype(float)
                fig_client.add_trace(go.Scatter(x=hh_axis[:len(v)], y=v, mode="lines",
                    name="Ridge résiduel", line=dict(color="#2196F3", dash="dash", width=2)))
            if lgbm_cols:
                v = row[sorted(lgbm_cols, key=lambda c: int(c.split("_")[-1]))].values.astype(float)
                fig_client.add_trace(go.Scatter(x=hh_axis[:len(v)], y=v, mode="lines",
                    name="LightGBM résiduel", line=dict(color="#4CAF50", dash="dash", width=2)))
            if cnn_cols:
                v = row[sorted(cnn_cols, key=lambda c: int(c.split("_")[-1]))].values.astype(float)
                fig_client.add_trace(go.Scatter(x=hh_axis[:len(v)], y=v, mode="lines",
                    name="CNN résiduel", line=dict(color="#E91E63", dash="dash", width=2)))

            seg_label = row.get("segment", "?")
            fig_client.update_layout(
                title=f"Client {pdl_sel} ({seg_label}) — Prédiction sur 2 jours",
                xaxis_title="Heure de la journée",
                yaxis_title="kWh / 30 min",
                height=440,
                hovermode="x unified",
                legend=dict(orientation="h", y=1.08),
                margin=dict(l=0, r=0, t=50, b=0),
            )
            st.plotly_chart(fig_client, use_container_width=True)

            # Métriques locales
            if y_cols and ridge_cols:
                y_flat = row[sorted(y_cols,    key=lambda c: int(c.split("_")[-1]))].values.astype(float)
                r_flat = row[sorted(ridge_cols, key=lambda c: int(c.split("_")[-1]))].values.astype(float)
                mae_r  = float(np.mean(np.abs(y_flat - r_flat)))
                rec_flat = row[sorted(recent_cols, key=lambda c: int(c.split("_")[-1]))].values.astype(float) if recent_cols else r_flat
                mae_b    = float(np.mean(np.abs(y_flat - rec_flat)))
                m1, m2, m3 = st.columns(3)
                with m1: st.metric("MAE Ridge",         f"{mae_r:.4f} kWh")
                with m2: st.metric("MAE Baseline récente", f"{mae_b:.4f} kWh")
                with m3: st.metric("Gain vs baseline",  f"{((mae_b - mae_r)/mae_b*100):.1f}%",
                                   delta_color="normal")

        else:
            st.info(
                "Aucun fichier d'exemples de prédiction trouvé. "
                f"Attendu : `{WINDOWS_FILE}`\n\n"
                "Ajoutez à la fin du notebook pour générer ce fichier :"
            )
            st.code(
                "# Exporter un échantillon de fenêtres avec prédictions\n"
                "sample_idx = np.random.default_rng(42).choice(idx_test, size=min(500, len(idx_test)), replace=False)\n"
                "sample_df  = windows.iloc[sample_idx][[\"pdl_id\", \"segment\", \"forecast_start\"]].copy()\n\n"
                "for i, col_idx in enumerate(sample_idx):\n"
                "    for h in range(HORIZON):\n"
                "        sample_df.loc[sample_idx[i], f'y_{h}']        = float(Y[col_idx, h])\n"
                "        sample_df.loc[sample_idx[i], f'b_recent_{h}'] = float(B_recent[col_idx, h])\n"
                "        sample_df.loc[sample_idx[i], f'b_weekly_{h}'] = float(B_weekly[col_idx, h])\n"
                "        pred_r = ridge_model.predict(X_tab[[col_idx]]).flatten()\n"
                "        sample_df.loc[sample_idx[i], f'ridge_{h}']    = float(np.clip(B_base[col_idx, h] + pred_r[h], 0, None))\n\n"
                "sample_df.to_csv(Results_DIR / 'prediction_windows_sample.csv', index=False)\n"
                "print('Fichier sauvegardé.')",
                language="python"
            )

    #  Tab 3 : Figures notebook 
    with tab_figs:
        st.subheader("Figures générées dans le notebook de prédiction")

        prediction_figs = {
            "Exemples de prédictions (6 clients)"      : FIG_PREDICTION_DIR / "fig_pred_01_exemples.png",
            "Comparaison WMAPE par segment"             : FIG_PREDICTION_DIR / "fig_pred_02_wmape_comparaison.png",
            "MAE intra-journalière"                     : FIG_PREDICTION_DIR / "fig_pred_03_mae_intraday.png",
            "Courbe de loss CNN"                        : FIG_PREDICTION_DIR / "fig_pred_04_cnn_loss.png",
            "Distribution des erreurs (boxplot)"        : FIG_PREDICTION_DIR / "fig_pred_05_erreurs_boxplot.png",
            "Historique + prédiction — client exemple"  : FIG_PREDICTION_DIR / "fig_pred_06_client_preview.png",
        }

        existing = {name: path for name, path in prediction_figs.items() if path.exists()}

        if existing:
            fig_sel = st.selectbox("Sélectionner une figure", list(existing.keys()))
            st.image(str(existing[fig_sel]), use_container_width=True)

            captions = {
                "Exemples de prédictions (6 clients)"     : "Comparaison réel vs prédit sur 6 clients du jeu de test.",
                "Comparaison WMAPE par segment"            : "Meilleur modèle par segment RS, RP, ALL selon le WMAPE.",
                "MAE intra-journalière"                    : "À quelles heures les modèles se trompent-ils le plus ?",
                "Courbe de loss CNN"                       : "Convergence du CNN 1D résiduel pendant l'entraînement.",
                "Distribution des erreurs (boxplot)"       : "Distribution des erreurs absolues, sans les outliers extrêmes.",
                "Historique + prédiction — client exemple" : "Historique récent et prédiction pour un client type.",
            }
            st.caption(captions.get(fig_sel, ""))
        else:
            st.warning(
                "Aucune figure trouvée dans `outputs/figures/prediction/`. "
                "Exécutez le notebook `03_prediction.ipynb` pour les générer."
            )

    #  Tab 4 : Limites 
    with tab_limits:
        st.subheader("Limites de la prédiction individuelle")
        st.markdown(
            """
            Même avec un modèle résiduel bien calibré, plusieurs défis subsistent :

            - **Résidences secondaires (RS)** : consommation souvent nulle ou très variable,
              rendant le MAPE instable. Le WMAPE atténue ce biais mais ne l'efface pas.
            - **Événements atypiques** (pannes, vacances non régulières) :
              non capturés par les features historiques.
            - **Dépendances longues** : le modèle Ridge ne capte pas les effets
              au-delà des 14 jours de fenêtre d'entrée.
            - **Généralisabilité** : le modèle est entraîné sur un seul dataset régional ;
              il peut ne pas se comporter de la même façon sur d'autres zones.

            **Piste d'amélioration :** un Transformer temporel (TFT) ou un modèle de type
            N-BEATS permettrait de capter des dépendances plus longues tout en restant
            interprétable.
            """
        )

# PAGE 6 : GENERATION
elif page == "🎨 Generer de nouveaux profils":
    st.title("🎨 Etape 3 — Générer des courbes de consommation synthétiques")

    st.markdown(
        '''<div class="story-box">
        La prévision nous dit <i>combien</i> un client consommera demain.
        Mais pour planifier le réseau à long terme — ou tester des scénarios futurs —
        on a besoin de <b>générer des courbes complètes</b> à pas de 30 minutes,
        réalistes et variées.<br><br>
        Un <b>CVAE</b> apprend la forme des courbes réelles et génère de nouvelles
        courbes synthétiques, conditionnées par le type de client et la saison.
        </div>''',
        unsafe_allow_html=True
    )

    gen_df = load_generated_curves()

    if gen_df is None:
        st.error(
            f"Fichier CSV introuvable ou invalide : {GENERATION_CSV_PATH}"
        )
        st.stop()

    tab_curves, tab_stats, tab_data, tab_limits = st.tabs(
        [
            "📈 Courbes générées",
            "📊 Analyse statistique",
            "📋 Données CSV",
            "⚠️ Limites"
        ]
    )

    with tab_curves:
        st.subheader("Visualisation des courbes synthétiques générées")

        c1, c2, c3 = st.columns(3)

        with c1:
            label_choice = st.selectbox(
                "Type de client",
                sorted(gen_df["label"].unique())
            )

        with c2:
            season_choice = st.selectbox(
                "Saison",
                sorted(gen_df["season"].unique())
            )

        filtered = gen_df[
            (gen_df["label"] == label_choice)
            & (gen_df["season"] == season_choice)
        ].copy()

        if filtered.empty:
            st.warning("Aucune courbe disponible pour ce filtre.")
        else:
            with c3:
                sample_choice = st.selectbox(
                    "Courbe générée",
                    sorted(filtered["sample_id"].unique())
                )

            curve = (
                filtered[filtered["sample_id"] == sample_choice]
                .sort_values("hh_index")
            )

            fig_curve = go.Figure()

            fig_curve.add_trace(go.Scatter(
                x=curve["time_h"],
                y=curve["puissance_w"],
                mode="lines+markers",
                name=f"Courbe {sample_choice}",
                line=dict(width=3),
                marker=dict(size=5),
            ))

            fig_curve.update_layout(
                title=f"Courbe générée — {label_choice} / {season_choice}",
                xaxis_title="Heure de la journée",
                yaxis_title="Puissance générée (W)",
                height=430,
                hovermode="x unified",
                margin=dict(l=0, r=0, t=40, b=0)
            )

            st.plotly_chart(fig_curve, use_container_width=True)

            total_kwh = curve["energy_kwh_step"].sum()
            max_power = curve["puissance_w"].max()
            mean_power = curve["puissance_w"].mean()

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric("Énergie journalière", f"{total_kwh:.2f} kWh")

            with m2:
                st.metric("Puissance maximale", f"{max_power:.0f} W")

            with m3:
                st.metric("Puissance moyenne", f"{mean_power:.0f} W")

            st.caption(
                "Chaque courbe générée contient 48 points : une journée complète "
                "au pas de 30 minutes."
            )

    with tab_stats:
        st.subheader("Analyse statistique des courbes générées")

        daily_gen = (
            gen_df
            .groupby(["sample_id", "label", "season"])
            .agg(
                total_kwh=("energy_kwh_step", "sum"),
                mean_power_w=("puissance_w", "mean"),
                max_power_w=("puissance_w", "max"),
                min_power_w=("puissance_w", "min"),
            )
            .reset_index()
        )

        st.markdown("#### Résumé par courbe générée")
        st.dataframe(daily_gen, use_container_width=True, hide_index=True)

        fig_box = px.box(
            daily_gen,
            x="label",
            y="total_kwh",
            color="season",
            title="Distribution de l’énergie journalière générée",
            labels={
                "label": "Type de client",
                "total_kwh": "Énergie journalière générée (kWh)",
                "season": "Saison"
            }
        )

        fig_box.update_layout(height=430)
        st.plotly_chart(fig_box, use_container_width=True)

        avg_profile = (
            gen_df
            .groupby(["label", "season", "time_h"])["puissance_w"]
            .mean()
            .reset_index()
        )

        fig_avg = px.line(
            avg_profile,
            x="time_h",
            y="puissance_w",
            color="label",
            line_dash="season",
            title="Profil moyen généré par type de client et saison",
            labels={
                "time_h": "Heure",
                "puissance_w": "Puissance moyenne générée (W)",
                "label": "Type de client",
                "season": "Saison"
            }
        )

        fig_avg.update_layout(height=430)
        st.plotly_chart(fig_avg, use_container_width=True)

    with tab_data:
        st.subheader("Contenu du fichier CSV généré")

        st.dataframe(
            gen_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            """
            **Colonnes du CSV :**

            - `sample_id` : identifiant de la courbe générée ;
            - `label` : type de client généré ;
            - `season` : saison conditionnelle ;
            - `hh_index` : pas demi-horaire de 0 à 47 ;
            - `time_h` : heure de la journée ;
            - `puissance_w` : puissance générée en watts ;
            - `energy_kwh_step` : énergie consommée sur le pas de 30 minutes.
            """
        )

    with tab_limits:
        st.subheader("Limites de la génération")

        st.markdown(
            """
            Les courbes générées sont utiles pour simuler des profils synthétiques,
            mais elles doivent être interprétées avec prudence.

            - Le CVAE apprend à partir des données disponibles : il peut reproduire leurs biais.
            - Une courbe visuellement réaliste n’est pas forcément statistiquement parfaite.
            - Il faut comparer les distributions réel/généré avec des tests statistiques.
            - Les profils générés ne remplacent pas des mesures terrain réelles.

            **Conclusion :** la génération est intéressante pour la simulation et l’augmentation
            de données, mais elle doit rester accompagnée d’une validation.
            """
        )