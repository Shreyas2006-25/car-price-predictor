"""Used Car Price Predictor - Streamlit GUI.   Run:  streamlit run app.py"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
st.set_page_config(page_title="CarWorth | Used Car Price Predictor", page_icon="🚗", layout="wide")

ACCENT, ACCENT2, MUTED, GRID = "#22d3a5", "#6c8cff", "#8b95b0", "rgba(255,255,255,.07)"

st.markdown(f"""
<style>
.block-container {{ padding-top: 2rem; max-width: 1250px; }}
#MainMenu, footer {{ visibility: hidden; }}
.hero h1 {{ font-size: 2.3rem; margin: 0; letter-spacing: -.02em; }}
.hero h1 span {{ background: linear-gradient(90deg,{ACCENT},{ACCENT2}); -webkit-background-clip: text;
                 -webkit-text-fill-color: transparent; }}
.hero p {{ color:{MUTED}; margin:.25rem 0 1.2rem; }}
.card {{ background:#141b33; border:1px solid rgba(255,255,255,.08); border-radius:18px; padding:1.4rem 1.6rem; }}
.price-card {{ background: radial-gradient(120% 140% at 0% 0%, rgba(34,211,165,.18), rgba(108,140,255,.10) 55%, #141b33);
               border:1px solid rgba(34,211,165,.35); border-radius:20px; padding:1.6rem 1.8rem; }}
.label {{ color:{MUTED}; font-size:.78rem; text-transform:uppercase; letter-spacing:.09em; }}
.price {{ font-size:3.3rem; font-weight:800; line-height:1.1; letter-spacing:-.02em; }}
.range {{ color:#c9d1e8; font-size:1.05rem; margin-top:.2rem; }}
.pill {{ display:inline-block; padding:.18rem .7rem; border-radius:99px; font-size:.78rem; margin:.5rem .35rem 0 0;
         background:rgba(255,255,255,.07); color:#d6dcf0; }}
.kpi {{ background:#141b33; border:1px solid rgba(255,255,255,.07); border-radius:14px; padding:.9rem 1.1rem; }}
.kpi b {{ display:block; font-size:1.45rem; margin-top:.15rem; }}
.note {{ color:{MUTED}; font-size:.82rem; }}
div[data-testid="stTabs"] button {{ font-size:1rem; }}
</style>
""", unsafe_allow_html=True)


# ---------- loading ----------
@st.cache_resource
def load_assets():
    model = joblib.load(ROOT / "model" / "price_model.joblib")
    meta = json.loads((ROOT / "model" / "meta.json").read_text())
    return model, meta


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data" / "cardekho.csv").drop_duplicates()


if not (ROOT / "model" / "price_model.joblib").exists():
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

model, meta = load_assets()
data = load_data()
opt = meta["options"]
REF = meta["ref_year"]
BAND = meta["error_band_80"]


def inr(v):
    v = float(v)
    return f"₹{v/1e7:.2f} Cr" if v >= 1e7 else f"₹{v/1e5:.2f} L"


def predict(rows: list[dict]) -> np.ndarray:
    df = pd.DataFrame(rows)
    df["Age"] = REF - df["Year"]
    df["Log_KM"] = np.log1p(df["KM_Driven"])
    return model.predict(df[["Brand", "Model", "Fuel", "Seller_Type", "Transmission", "Owner", "Age", "KM_Driven", "Log_KM"]])


def style(fig, h=360):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1e8"),
                      xaxis=dict(gridcolor=GRID, zeroline=False), yaxis=dict(gridcolor=GRID, zeroline=False),
                      legend=dict(orientation="h", y=1.12, x=0))
    return fig


# ---------- header ----------
st.markdown(f"""<div class="hero"><h1>🚗 Car<span>Worth</span></h1>
<p>Machine-learning price estimates for second-hand cars, trained on {meta['rows']:,} real CarDekho listings.</p></div>""",
            unsafe_allow_html=True)

tab_pred, tab_market, tab_model = st.tabs(["💰 Price estimate", "📊 Market explorer", "🧠 Model insights"])

# =====================================================================
with tab_pred:
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        st.markdown("#### Describe the car")
        brand = st.selectbox("Brand", opt["brands"], index=opt["brands"].index("Maruti") if "Maruti" in opt["brands"] else 0)
        models = opt["models_by_brand"][brand]
        model_name = st.selectbox("Model / variant", models)
        c1, c2 = st.columns(2)
        year = c1.slider("Year of purchase", opt["year_min"], opt["year_max"], 2015)
        km = c2.number_input("Kilometres driven", 0, 500_000, 60_000, step=5_000)
        c3, c4 = st.columns(2)
        fuel = c3.selectbox("Fuel", opt["fuel"], index=opt["fuel"].index("Petrol"))
        trans = c4.selectbox("Transmission", opt["transmission"], index=opt["transmission"].index("Manual"))
        c5, c6 = st.columns(2)
        seller = c5.selectbox("Seller type", opt["seller"], index=opt["seller"].index("Individual"))
        owner = c6.selectbox("Ownership", opt["owner"])

    car = dict(Brand=brand, Model=model_name, Year=year, KM_Driven=km, Fuel=fuel,
               Seller_Type=seller, Transmission=trans, Owner=owner)
    est = float(predict([car])[0])
    lo, hi = est * (1 - BAND), est * (1 + BAND)

    with right:
        st.markdown(f"""<div class="price-card">
            <div class="label">Estimated resale value</div>
            <div class="price">{inr(est)}</div>
            <div class="range">Likely range <b>{inr(lo)} – {inr(hi)}</b></div>
            <span class="pill">{year} · {REF-year} yrs old</span><span class="pill">{km:,} km</span>
            <span class="pill">{fuel}</span><span class="pill">{trans}</span><span class="pill">{owner}</span>
            </div>""", unsafe_allow_html=True)
        st.caption(f"Range = the error band within which 80% of held-out test predictions fell (±{BAND:.0%}).")

        # price position vs comparable cars
        peers = data[(data.Brand == brand) & (data.Year.between(year - 2, year + 2))]
        exact = peers[peers.Model == model_name]
        k1, k2, k3 = st.columns(3)
        k1.markdown(f'<div class="kpi"><span class="label">Exact ₹ value</span><b>₹{est:,.0f}</b></div>', unsafe_allow_html=True)
        if len(peers) >= 5:
            pct = (peers.Selling_Price < est).mean() * 100
            k2.markdown(f'<div class="kpi"><span class="label">{brand} ±2 yr median</span><b>{inr(peers.Selling_Price.median())}</b></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="kpi"><span class="label">Price percentile</span><b>{pct:.0f}th</b></div>', unsafe_allow_html=True)
        else:
            k2.markdown('<div class="kpi"><span class="label">Comparable cars</span><b>few</b></div>', unsafe_allow_html=True)
            k3.markdown('<div class="kpi"><span class="label">Percentile</span><b>–</b></div>', unsafe_allow_html=True)

    st.markdown("---")
    g1, g2 = st.columns(2, gap="large")

    # depreciation curve: same car at different purchase years (km scaled to the same yearly usage)
    with g1:
        st.markdown("##### 📉 How value changes with age")
        per_year = km / max(REF - year, 1)
        yrs = list(range(opt["year_min"], opt["year_max"] + 1))
        rows = [{**car, "Year": y, "KM_Driven": int(per_year * (REF - y))} for y in yrs]
        pr = predict(rows)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yrs, y=pr / 1e5, mode="lines", line=dict(color=ACCENT2, width=3),
                                 fill="tozeroy", fillcolor="rgba(108,140,255,.12)", name="Estimated value"))
        fig.add_trace(go.Scatter(x=[year], y=[est / 1e5], mode="markers", marker=dict(size=14, color=ACCENT,
                                 line=dict(color="white", width=2)), name="Your car"))
        fig.update_layout(xaxis_title="Year of purchase", yaxis_title="Value (₹ lakh)")
        st.plotly_chart(style(fig), width="stretch")
        st.markdown(f'<p class="note">Same car at the same ~{per_year:,.0f} km/year usage, bought in different years.</p>', unsafe_allow_html=True)

    # km sensitivity
    with g2:
        st.markdown("##### 🛣️ How mileage affects the price")
        kms = np.linspace(5_000, 250_000, 40).astype(int)
        pk = predict([{**car, "KM_Driven": int(v)} for v in kms])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=kms, y=pk / 1e5, mode="lines", line=dict(color=ACCENT, width=3),
                                 fill="tozeroy", fillcolor="rgba(34,211,165,.10)", name="Estimated value"))
        fig.add_trace(go.Scatter(x=[km], y=[est / 1e5], mode="markers", marker=dict(size=14, color="#ffb454",
                                 line=dict(color="white", width=2)), name="Your car"))
        fig.update_layout(xaxis_title="Kilometres driven", yaxis_title="Value (₹ lakh)")
        st.plotly_chart(style(fig), width="stretch")

    with st.expander(f"🔎 Similar listings in the dataset ({len(exact) if len(exact) else len(peers)})"):
        sim = exact if len(exact) >= 3 else peers
        st.caption("Same model" if len(exact) >= 3 else f"Same brand, bought {year-2}–{year+2}")
        st.dataframe(sim.sort_values("Year", ascending=False).head(25)
                     .rename(columns={"Selling_Price": "Price (₹)", "KM_Driven": "KM"}),
                     width="stretch", hide_index=True)

# =====================================================================
with tab_market:
    st.markdown("#### Explore the CarDekho market data")
    m1, m2, m3 = st.columns(3)
    sel_brands = m1.multiselect("Brands", opt["brands"], default=["Maruti", "Hyundai", "Honda", "Toyota"])
    sel_fuel = m2.multiselect("Fuel", opt["fuel"], default=["Petrol", "Diesel"])
    yr = m3.slider("Year range", opt["year_min"], opt["year_max"], (2010, opt["year_max"]))
    d = data[data.Brand.isin(sel_brands) & data.Fuel.isin(sel_fuel) & data.Year.between(*yr)
             & (data.KM_Driven < 500_000)]
    if d.empty:
        st.info("No listings match these filters.")
    else:
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("##### Price by brand")
            fig = go.Figure()
            for br, g in d.groupby("Brand"):
                fig.add_trace(go.Box(y=g.Selling_Price / 1e5, name=br, boxpoints=False))
            fig.update_layout(yaxis_title="Price (₹ lakh)", showlegend=False)
            st.plotly_chart(style(fig, 380), width="stretch")
        with b:
            st.markdown("##### Price vs kilometres")
            fig = go.Figure()
            for br, g in d.groupby("Brand"):
                fig.add_trace(go.Scattergl(x=g.KM_Driven, y=g.Selling_Price / 1e5, mode="markers", name=br,
                                           marker=dict(size=6, opacity=.6)))
            fig.update_layout(xaxis_title="Kilometres driven", yaxis_title="Price (₹ lakh)")
            st.plotly_chart(style(fig, 380), width="stretch")
        med = d.groupby("Year").Selling_Price.median().reset_index()
        st.markdown("##### Median price by year")
        fig = go.Figure(go.Bar(x=med.Year, y=med.Selling_Price / 1e5, marker_color=ACCENT2))
        fig.update_layout(yaxis_title="Median price (₹ lakh)")
        st.plotly_chart(style(fig, 300), width="stretch")
        st.caption(f"{len(d):,} listings shown.")

# =====================================================================
with tab_model:
    st.markdown("#### How good is the model?")
    best = meta["best_model"]
    bm = meta["metrics"][best]
    q1, q2, q3, q4 = st.columns(4)
    q1.markdown(f'<div class="kpi"><span class="label">Selected model</span><b>{best}</b></div>', unsafe_allow_html=True)
    q2.markdown(f'<div class="kpi"><span class="label">Test R²</span><b>{bm["test_r2"]:.2f}</b></div>', unsafe_allow_html=True)
    q3.markdown(f'<div class="kpi"><span class="label">Mean abs. error</span><b>₹{bm["test_mae"]/1e3:,.0f}k</b></div>', unsafe_allow_html=True)
    q4.markdown(f'<div class="kpi"><span class="label">Mean % error</span><b>{bm["test_mape"]:.0%}</b></div>', unsafe_allow_html=True)
    st.write("")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("##### Predicted vs actual (held-out test set)")
        t = meta["test_sample"]
        act, prd = np.array(t["actual"]) / 1e5, np.array(t["predicted"]) / 1e5
        mx = max(act.max(), prd.max())
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=act, y=prd, mode="markers", marker=dict(color=ACCENT, size=6, opacity=.6), name="Cars"))
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color="#ffb454", dash="dash"), name="Perfect"))
        fig.update_layout(xaxis_title="Actual (₹ lakh)", yaxis_title="Predicted (₹ lakh)")
        st.plotly_chart(style(fig, 380), width="stretch")
    with b:
        st.markdown("##### Model comparison (5-fold CV error, lower is better)")
        names = list(meta["metrics"])
        fig = go.Figure(go.Bar(x=[meta["metrics"][n]["cv_mape"] * 100 for n in names], y=names, orientation="h",
                               marker_color=[ACCENT if n == best else "#3a4570" for n in names],
                               text=[f'{meta["metrics"][n]["cv_mape"]:.1%}' for n in names], textposition="outside"))
        fig.update_layout(xaxis_title="Mean % error", xaxis_range=[0, 36], yaxis=dict(autorange="reversed"))
        st.plotly_chart(style(fig, 380), width="stretch")
    st.info(
        "**Reading these numbers honestly:** the dataset has no engine size, power, trim, or condition, and "
        "listings are asking prices, so even a good model has a typical error around 25%. Treat the estimate as a "
        "starting point for negotiation, not a quote.")
    st.caption(f"Trained {meta['trained_at']} on {meta['rows']:,} de-duplicated listings (CarDekho, via Kaggle).")
