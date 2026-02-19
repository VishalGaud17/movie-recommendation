# app.py  –  streamlit run app.py

import pickle
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [data-testid="stAppViewContainer"] { background:#0a0a0f; color:#f0f0f0; }
[data-testid="stHeader"] { background:transparent; }
* { font-family:'DM Sans',sans-serif; }

/* hero */
.hero {
    background:linear-gradient(135deg,#1a0533 0%,#0d1b4b 50%,#0a2e2e 100%);
    border-radius:20px; padding:3rem 2.5rem 2.2rem;
    margin-bottom:2rem; position:relative; overflow:hidden;
    border:1px solid rgba(255,255,255,0.06);
}
.hero::before {
    content:''; position:absolute; inset:0;
    background:radial-gradient(ellipse at 70% 50%,rgba(139,92,246,.18) 0%,transparent 60%),
               radial-gradient(ellipse at 20% 80%,rgba(6,182,212,.12) 0%,transparent 50%);
}
.hero-title {
    font-family:'Bebas Neue',sans-serif; font-size:4.5rem; line-height:1;
    background:linear-gradient(90deg,#c084fc,#67e8f9,#f0abfc);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin:0 0 .4rem; position:relative;
}
.hero-sub { color:rgba(255,255,255,.5); font-size:.95rem; font-weight:300; position:relative; }
.badge {
    display:inline-block;
    background:rgba(139,92,246,.2); border:1px solid rgba(139,92,246,.4);
    color:#c084fc; font-size:.7rem; font-weight:700; letter-spacing:.12em;
    padding:.25rem .75rem; border-radius:999px; margin-bottom:1rem; position:relative;
}

/* selectbox */
.stSelectbox>label { color:rgba(255,255,255,.65)!important; font-size:.8rem; font-weight:500; letter-spacing:.05em; }
[data-testid="stSelectbox"]>div>div {
    background:#13131f!important; border:1px solid rgba(139,92,246,.35)!important;
    border-radius:12px!important; color:#f0f0f0!important;
}

/* button */
.stButton>button {
    background:linear-gradient(135deg,#7c3aed,#2563eb)!important;
    color:#fff!important; font-weight:700!important; border:none!important;
    border-radius:12px!important; padding:.7rem 0!important;
    font-size:.95rem!important; width:100%!important;
    box-shadow:0 4px 20px rgba(124,58,237,.4)!important;
    transition:opacity .2s,transform .15s!important;
}
.stButton>button:hover { opacity:.88!important; transform:translateY(-1px)!important; }

/* selected movie info */
.info-card {
    background:linear-gradient(135deg,#1a1a2e,#16213e);
    border:1px solid rgba(99,102,241,.3); border-radius:16px;
    padding:1.4rem 1.6rem; margin-bottom:1.8rem;
}
.info-card-title { font-family:'Bebas Neue',sans-serif; font-size:2rem; color:#a5b4fc; margin:0 0 .8rem; }
.info-pill {
    display:inline-flex; align-items:center; gap:.4rem;
    background:rgba(255,255,255,.06); border-radius:999px;
    padding:.3rem .85rem; font-size:.78rem; color:rgba(255,255,255,.55);
    margin:.2rem .25rem .2rem 0;
}
.info-pill span { color:#c4b5fd; font-weight:600; }

/* results heading */
.results-heading {
    font-family:'Bebas Neue',sans-serif; font-size:1.6rem;
    color:rgba(255,255,255,.8); letter-spacing:.04em; margin:0 0 1rem;
}

/* result card */
.rec-card {
    background:#111120; border:1px solid rgba(255,255,255,.06);
    border-radius:14px; padding:1rem 1.25rem;
    margin-bottom:.65rem; display:flex; align-items:center; gap:1rem;
    position:relative; overflow:hidden;
}
.rec-card::before {
    content:''; position:absolute; left:0; top:0; bottom:0; width:3px;
    background:linear-gradient(180deg,#8b5cf6,#06b6d4); border-radius:3px 0 0 3px;
}
.rec-rank { font-family:'Bebas Neue',sans-serif; font-size:2rem; color:rgba(139,92,246,.35); min-width:2.8rem; text-align:center; line-height:1; }
.rec-body { flex:1; }
.rec-title { font-size:.98rem; font-weight:700; color:#e2e8f0; margin-bottom:.2rem; }
.rec-meta  { font-size:.74rem; color:rgba(255,255,255,.38); }
.rec-meta b { color:rgba(255,255,255,.58); }
.score-badge {
    background:linear-gradient(135deg,rgba(139,92,246,.2),rgba(6,182,212,.15));
    border:1px solid rgba(139,92,246,.3); border-radius:999px;
    padding:.3rem .75rem; font-size:.74rem; font-weight:700; color:#a5b4fc; white-space:nowrap;
}

/* stat strip */
.stat-strip {
    display:flex; gap:1rem; background:#0f0f1a; border-radius:12px;
    padding:.9rem 1.2rem; margin-top:2rem; border:1px solid rgba(255,255,255,.05);
}
.stat { flex:1; text-align:center; }
.stat-val { font-family:'Bebas Neue',sans-serif; font-size:1.6rem; color:#c084fc; }
.stat-lbl { font-size:.68rem; color:rgba(255,255,255,.32); letter-spacing:.08em; text-transform:uppercase; }

hr { border-color:rgba(255,255,255,.06)!important; }
</style>
""", unsafe_allow_html=True)


# ── Load pkl artifacts ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🎬  Loading model artifacts…")
def load_artifacts():
    with open('movies_df.pkl',    'rb') as f: df           = pickle.load(f)
    with open('tfidf_matrix.pkl', 'rb') as f: tfidf_matrix = pickle.load(f)
    with open('indices.pkl',      'rb') as f: indices      = pickle.load(f)
    return df, tfidf_matrix, indices


# ── Recommend ─────────────────────────────────────────────────────────────────
def recommend(title, df, tfidf_matrix, indices, n=10):
    if title not in indices:
        return None, None
    idx        = indices[title]
    movie_info = df.iloc[idx]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    top_idx    = sim_scores.argsort()[::-1][1:n + 1]
    results    = pd.DataFrame({
        "title":            df['title'].iloc[top_idx].values,
        "similarity_score": sim_scores[top_idx],
        "cast":             df['cast'].iloc[top_idx].values,
        "director":         df['director'].iloc[top_idx].values,
        "genres":           df['genres'].iloc[top_idx].values,
    }).reset_index(drop=True)
    return movie_info, results


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="badge">✦ AI POWERED</div>
  <div class="hero-title">CineMatch</div>
  <div class="hero-sub">Content-based movie recommendations · TF-IDF &amp; Cosine Similarity</div>
</div>
""", unsafe_allow_html=True)

try:
    df, tfidf_matrix, indices = load_artifacts()
    movie_list = sorted(df['title'].dropna().unique().tolist())

    # ── Controls ───────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        selected = st.selectbox("🔍  Search Movie Title", movie_list)
    with col2:
        n_recs = st.selectbox("Results", [5, 10, 15, 20], index=1)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Recommend →")

    st.markdown("<hr>", unsafe_allow_html=True)

    if run:
        movie_info, results = recommend(selected, df, tfidf_matrix, indices, n=n_recs)

        if results is None:
            st.error("Movie not found in the dataset.")
        else:
            # ── Selected movie card ────────────────────────────────────────────
            genres_str   = str(movie_info.get('genres',   '—'))[:80]
            cast_str     = str(movie_info.get('cast',     '—'))[:80]
            director_str = str(movie_info.get('director', '—'))[:60]

            st.markdown(f"""
            <div class="info-card">
              <div class="info-card-title">▸ {movie_info['title']}</div>
              <div>
                <div class="info-pill">🎭 Director <span>{director_str}</span></div>
                <div class="info-pill">🎬 Cast <span>{cast_str}…</span></div>
                <div class="info-pill">🎞️ Genres <span>{genres_str}</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Recommendation cards (2-column) ───────────────────────────────
            st.markdown(f'<div class="results-heading">Top {n_recs} Similar Movies</div>',
                        unsafe_allow_html=True)

            left_col, right_col = st.columns(2)
            for i, row in results.iterrows():
                score_pct = f"{row['similarity_score'] * 100:.1f}%"
                director  = str(row['director'])[:40]
                cast_val  = str(row['cast'])[:50]
                genres_v  = str(row['genres'])[:40]
                card = f"""
                <div class="rec-card">
                  <div class="rec-rank">{i+1:02d}</div>
                  <div class="rec-body">
                    <div class="rec-title">{row['title']}</div>
                    <div class="rec-meta">
                      <b>Dir:</b> {director} &nbsp;·&nbsp;
                      <b>Cast:</b> {cast_val}… &nbsp;·&nbsp;
                      <b>Genres:</b> {genres_v}
                    </div>
                  </div>
                  <div class="score-badge">⚡ {score_pct}</div>
                </div>"""
                (left_col if i % 2 == 0 else right_col).markdown(card, unsafe_allow_html=True)

    # ── Stat strip ─────────────────────────────────────────────────────────────
    directors = df['director'].nunique() if 'director' in df.columns else "—"
    st.markdown(f"""
    <div class="stat-strip">
      <div class="stat"><div class="stat-val">{len(df):,}</div><div class="stat-lbl">Movies</div></div>
      <div class="stat"><div class="stat-val">{directors:,}</div><div class="stat-lbl">Directors</div></div>
      <div class="stat"><div class="stat-val">TF-IDF</div><div class="stat-lbl">Algorithm</div></div>
      <div class="stat"><div class="stat-val">50K</div><div class="stat-lbl">Max Features</div></div>
      <div class="stat"><div class="stat-val">1–2</div><div class="stat-lbl">N-Gram Range</div></div>
    </div>
    """, unsafe_allow_html=True)

except FileNotFoundError as e:
    st.error(
        f"**Missing file:** `{e.filename}`\n\n"
        "Make sure these pkl files are in the same folder as `app.py`:\n"
        "```\nmovies_df.pkl\ntfidf_matrix.pkl\nindices.pkl\n```\n"
        "Generate them by running your notebook top-to-bottom (the pickle.dump cells at the end)."
    )
