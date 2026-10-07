"""
House Price Estimator - Streamlit web app
-----------------------------------------
Tab 1: estimate the price of a house (with a likely range) and download it.
Tab 2: enter a budget and see which houses you can afford.

Prices are shown in Indian Rupees (INR).

HOW PRICES ARE CALCULATED
The training dataset has almost no relationship between house features and
price (the trained model scores R2 ~ 0, no better than guessing the average).
So estimates here come from a transparent market-rate formula:

    price = area x rate per sq ft
            x location factor x condition factor x age factor
            x garage factor x bathroom factor x floor factor

Every factor is an assumption you can edit in the sidebar. The trained ML
model is still shown in the app for reference, with its real accuracy.

Install:
    python -m pip install streamlit pandas numpy scikit-learn joblib fpdf2

Run (from the folder with this file, the CSV and the .joblib model):
    python -m streamlit run app.py
"""

import itertools
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    from fpdf import FPDF

    HAS_PDF = True
except ImportError:  # PDF download is optional
    HAS_PDF = False

DATA_PATH = "House Price Prediction Dataset.csv"
MODEL_PATH = "house_price_model.joblib"
METRICS_PATH = "model_metrics.json"
CURRENT_YEAR = datetime.now().year

# ----------------------------------------------------------------------------
# Pricing assumptions (editable in the sidebar where marked)
# ----------------------------------------------------------------------------
DEFAULT_RATE_PER_SQFT = 4000  # Rs per sq ft for an average house, mid-range city
DEFAULT_LOCATION_FACTORS = {"Downtown": 1.25, "Urban": 1.15, "Suburban": 1.00, "Rural": 0.70}
DEFAULT_CONDITION_FACTORS = {"Excellent": 1.15, "Good": 1.00, "Fair": 0.90, "Poor": 0.75}
GARAGE_FACTOR = 1.05  # house with a garage
AGE_DEPRECIATION_PER_YEAR = 0.005  # 0.5% per year of age ...
MAX_AGE_DEPRECIATION = 0.25  # ... capped at 25%
BATHROOM_BONUS = 0.03  # +3% for each bathroom beyond the first
FLOOR_BONUS = 0.02  # +2% for each floor beyond the first
DEFAULT_RANGE_PCT = 10  # +/- % shown as the likely price range

st.set_page_config(page_title="House Price Estimator", page_icon="🏠", layout="centered")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering used by the trained ML model (must match training)."""
    df = df.copy()
    df["HouseAge"] = CURRENT_YEAR - df["YearBuilt"]
    df["TotalRooms"] = df["Bedrooms"] + df["Bathrooms"]
    df["AreaPerBedroom"] = df["Area"] / df["Bedrooms"].replace(0, np.nan)
    df["AreaPerBedroom"] = df["AreaPerBedroom"].fillna(df["Area"])
    return df


def format_inr(amount: float) -> str:
    """Indian digit grouping, e.g. 12,34,56,789."""
    amount = int(round(amount))
    s = str(abs(amount))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    return ("-" if amount < 0 else "") + "₹" + s


def inr_in_words(amount: float) -> str:
    if amount >= 1e7:
        return f"₹{amount / 1e7:.2f} Crore"
    if amount >= 1e5:
        return f"₹{amount / 1e5:.2f} Lakh"
    return format_inr(amount)


def compute_factors(df, loc_f, cond_f):
    """Return a DataFrame of price multipliers, one column per factor."""
    f = pd.DataFrame(index=df.index)
    f["Location"] = df["Location"].map(loc_f).fillna(1.0)
    f["Condition"] = df["Condition"].map(cond_f).fillna(1.0)
    age = (CURRENT_YEAR - df["YearBuilt"]).clip(lower=0)
    f["Age"] = 1 - (age * AGE_DEPRECIATION_PER_YEAR).clip(upper=MAX_AGE_DEPRECIATION)
    has_garage = df["Garage"].astype(str).str.lower().isin(["yes", "true", "1"])
    f["Garage"] = np.where(has_garage, GARAGE_FACTOR, 1.0)
    f["Bathrooms"] = 1 + BATHROOM_BONUS * (df["Bathrooms"] - 1).clip(lower=0)
    f["Floors"] = 1 + FLOOR_BONUS * (df["Floors"] - 1).clip(lower=0)
    return f


def estimate_prices(df, rate, loc_f, cond_f):
    factors = compute_factors(df, loc_f, cond_f)
    return df["Area"] * rate * factors.prod(axis=1), factors


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


@st.cache_data
def load_options():
    df = pd.read_csv(DATA_PATH)
    return {
        "locations": sorted(df["Location"].dropna().unique().tolist()),
        "conditions": sorted(df["Condition"].dropna().unique().tolist()),
        "garages": sorted(df["Garage"].dropna().unique().tolist()),
        "bedrooms": sorted(df["Bedrooms"].dropna().astype(int).unique().tolist()),
        "bathrooms": sorted(df["Bathrooms"].dropna().astype(int).unique().tolist()),
        "floors": sorted(df["Floors"].dropna().astype(int).unique().tolist()),
        "area_min": int(df["Area"].min()),
        "area_max": int(df["Area"].max()),
        "area_median": int(df["Area"].median()),
        "price_median": float(df["Price"].median()),
        "year_min": int(df["YearBuilt"].min()),
        "year_max": int(df["YearBuilt"].max()),
    }


def make_pdf(result: dict) -> bytes:
    """One-page PDF summary of an estimate (uses 'Rs.' - core PDF fonts lack the rupee sign)."""

    def rs(x):
        return format_inr(x).replace("₹", "Rs. ")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "House Price Estimate", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, datetime.now().strftime("Generated on %d %B %Y, %H:%M"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Property details", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for k, v in result["inputs"].items():
        pdf.cell(60, 7, k)
        pdf.cell(0, 7, str(v), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Estimate", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, rs(result["price"]), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Likely range: {rs(result['low'])} to {rs(result['high'])}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Approx. price per sq ft: {rs(result['price'] / result['inputs']['Area (sq ft)'])}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "How it was calculated", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Base: area x Rs. {result['rate']:,.0f} per sq ft = {rs(result['base'])}",
             new_x="LMARGIN", new_y="NEXT")
    for k, v in result["factors"].items():
        pdf.cell(0, 7, f"  x {k} factor: {v:.3f}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "This is an estimate from a market-rate formula with adjustable "
                         "assumptions, not an official valuation.")
    return bytes(pdf.output())


# ----------------------------------------------------------------------------
# Load data and model
# ----------------------------------------------------------------------------
try:
    opt = load_options()
except FileNotFoundError as e:
    st.error(f"File not found: {e.filename}\n\nKeep app.py and the CSV in the same folder.")
    st.stop()

model = load_model()
metrics = load_metrics()

# ----------------------------------------------------------------------------
# Sidebar: assumptions
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Pricing assumptions")
    rate = st.number_input(
        "Market rate (₹ per sq ft)",
        min_value=500,
        max_value=100000,
        value=DEFAULT_RATE_PER_SQFT,
        step=250,
        help="Rate for an average house in your city. Set it to match your local market.",
    )
    range_pct = st.slider("Price range (± %)", 5, 30, DEFAULT_RANGE_PCT,
                          help="Half-width of the likely price range.")

    with st.expander("Location factors"):
        loc_f = {
            loc: st.number_input(loc, 0.3, 3.0, float(DEFAULT_LOCATION_FACTORS.get(loc, 1.0)),
                                 step=0.05, key=f"loc_{loc}")
            for loc in opt["locations"]
        }
    with st.expander("Condition factors"):
        cond_f = {
            c: st.number_input(c, 0.3, 3.0, float(DEFAULT_CONDITION_FACTORS.get(c, 1.0)),
                               step=0.05, key=f"cond_{c}")
            for c in opt["conditions"]
        }
    st.caption(
        f"Also applied: age (−{AGE_DEPRECIATION_PER_YEAR * 100:.1f}% per year, max "
        f"−{MAX_AGE_DEPRECIATION * 100:.0f}%), garage (+{(GARAGE_FACTOR - 1) * 100:.0f}%), "
        f"+{BATHROOM_BONUS * 100:.0f}% per extra bathroom, +{FLOOR_BONUS * 100:.0f}% per extra floor."
    )

# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.title("🏠 House Price Estimator")
tab_est, tab_budget = st.tabs(["Estimate a price", "Find houses in my budget"])

# ============================ TAB 1: ESTIMATE ===============================
with tab_est:
    st.write("Enter the details of the house to get an estimated price in Indian Rupees.")

    with st.form("house_form"):
        col1, col2 = st.columns(2)
        with col1:
            area = st.number_input("Area (sq ft)", min_value=100,
                                   max_value=max(opt["area_max"] * 2, 10000),
                                   value=opt["area_median"], step=50)
            bedrooms = st.selectbox("Bedrooms", opt["bedrooms"])
            bathrooms = st.selectbox("Bathrooms", opt["bathrooms"])
            floors = st.selectbox("Floors", opt["floors"])
        with col2:
            year_built = st.number_input("Year built", min_value=1800, max_value=CURRENT_YEAR,
                                         value=min(max(2000, opt["year_min"]), CURRENT_YEAR),
                                         step=1)
            location = st.selectbox("Location", opt["locations"])
            condition = st.selectbox("Condition", opt["conditions"])
            garage = st.selectbox("Garage", opt["garages"])
        submitted = st.form_submit_button("Estimate Price", width="stretch")

    if submitted:
        house = pd.DataFrame([{
            "Area": area, "Bedrooms": bedrooms, "Bathrooms": bathrooms, "Floors": floors,
            "YearBuilt": year_built, "Location": location, "Condition": condition,
            "Garage": garage,
        }])
        price_s, factors = estimate_prices(house, rate, loc_f, cond_f)
        price = float(price_s.iloc[0])
        ml_price = None
        if model is not None:
            raw = float(model.predict(add_features(house).drop(columns=["YearBuilt"]))[0])
            ml_price = max(raw, 0) * rate * opt["area_median"] / opt["price_median"]
        st.session_state["est"] = {
            "price": price,
            "low": price * (1 - range_pct / 100),
            "high": price * (1 + range_pct / 100),
            "range_pct": range_pct,
            "rate": rate,
            "base": float(area * rate),
            "factors": factors.iloc[0].to_dict(),
            "ml_price": ml_price,
            "inputs": {
                "Area (sq ft)": area, "Bedrooms": bedrooms, "Bathrooms": bathrooms,
                "Floors": floors, "Year built": year_built, "Location": location,
                "Condition": condition, "Garage": garage,
            },
        }

    est = st.session_state.get("est")
    if est:
        st.success("Estimation complete")
        st.metric("Estimated price", format_inr(est["price"]))
        st.write(f"**Likely range (±{est['range_pct']}%):** "
                 f"{format_inr(est['low'])} to {format_inr(est['high'])}")
        st.write(f"**In words:** {inr_in_words(est['price'])}  \n"
                 f"**Approx. price per sq ft:** "
                 f"{format_inr(est['price'] / est['inputs']['Area (sq ft)'])}")

        with st.expander("How this was calculated"):
            rows = [("Base: area × market rate", f"{format_inr(est['base'])}")]
            rows += [(f"× {k} factor", f"{v:.3f}") for k, v in est["factors"].items()]
            rows.append(("Estimated price", format_inr(est["price"])))
            st.table(pd.DataFrame(rows, columns=["Step", "Value"]))

        if est["ml_price"] is not None:
            with st.expander("What the trained ML model says (reference only)"):
                st.write(f"ML model output: **{format_inr(est['ml_price'])}**")
                if metrics:
                    st.caption(
                        f"Model: {metrics['best_model']}. Test R² = {metrics['test_r2']:.3f} "
                        f"(baseline that always guesses the average: {metrics['baseline_r2']:.3f}). "
                        "An R² near 0 means the dataset's features barely explain price, so "
                        "the formula above is used for the main estimate."
                    )

        # Downloads
        row = {**est["inputs"], "Estimated price (INR)": round(est["price"]),
               "Range low (INR)": round(est["low"]), "Range high (INR)": round(est["high"]),
               "Market rate (INR/sq ft)": est["rate"]}
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Download CSV", pd.DataFrame([row]).to_csv(index=False).encode(),
                           file_name="house_price_estimate.csv", mime="text/csv",
                           width="stretch")
        if HAS_PDF:
            d2.download_button("⬇ Download PDF", make_pdf(est),
                               file_name="house_price_estimate.pdf", mime="application/pdf",
                               width="stretch")
        else:
            d2.caption("Install fpdf2 to enable PDF download.")

        st.caption("This is an estimate from a market-rate formula, not an official valuation.")

# ============================ TAB 2: BUDGET =================================
with tab_budget:
    st.write("Enter your budget and preferences to see what you can afford.")

    with st.form("budget_form"):
        b1, b2 = st.columns(2)
        with b1:
            budget = st.number_input("Budget (₹)", min_value=500000, max_value=1000000000,
                                     value=10000000, step=500000)
            b_location = st.selectbox("Location", opt["locations"], key="b_loc")
            b_condition = st.selectbox("Condition", opt["conditions"],
                                       index=min(1, len(opt["conditions"]) - 1), key="b_cond")
        with b2:
            min_bed = st.selectbox("Minimum bedrooms", opt["bedrooms"], key="b_bed")
            min_bath = st.selectbox("Minimum bathrooms", opt["bathrooms"], key="b_bath")
            b_garage = st.selectbox("Garage", ["Any"] + opt["garages"], key="b_gar")
            b_year = st.number_input("Year built (approx.)", min_value=1800,
                                     max_value=CURRENT_YEAR, value=2010, step=1, key="b_year")
        go = st.form_submit_button("Find houses", width="stretch")

    if go:
        areas = np.arange(max(opt["area_min"], 100), opt["area_max"] + 1, 100)
        combos = itertools.product(
            areas,
            [b for b in opt["bedrooms"] if b >= min_bed],
            [b for b in opt["bathrooms"] if b >= min_bath],
            opt["floors"],
            opt["garages"] if b_garage == "Any" else [b_garage],
        )
        grid = pd.DataFrame(combos, columns=["Area", "Bedrooms", "Bathrooms", "Floors", "Garage"])
        grid["YearBuilt"] = b_year
        grid["Location"] = b_location
        grid["Condition"] = b_condition
        prices, _ = estimate_prices(grid, rate, loc_f, cond_f)
        grid["Price"] = prices
        st.session_state["bud"] = {"grid": grid, "budget": budget}

    bud = st.session_state.get("bud")
    if bud:
        grid, budget_val = bud["grid"], bud["budget"]
        st.write(f"**Your budget:** {format_inr(budget_val)} ({inr_in_words(budget_val)})")
        ok = grid[grid["Price"] <= budget_val]

        if ok.empty:
            st.warning(
                f"No house in the searched range fits this budget. The cheapest option "
                f"costs about {format_inr(grid['Price'].min())}. Try a higher budget, a lower "
                "market rate in the sidebar, a different location, or fewer minimums."
            )
        else:
            # Best option (largest area) for every bedroom/bathroom layout
            best = (ok.sort_values(["Area", "Price"], ascending=[False, False])
                      .groupby(["Bedrooms", "Bathrooms"], as_index=False).head(1)
                      .sort_values(["Bedrooms", "Bathrooms"]))
            top = best.loc[best["Area"].idxmax()]
            st.success(f"Largest house you can afford: **{int(top['Area'])} sq ft** "
                       f"({int(top['Bedrooms'])} bed, {int(top['Bathrooms'])} bath) "
                       f"at about {inr_in_words(top['Price'])}.")

            show = pd.DataFrame({
                "Bedrooms": best["Bedrooms"].astype(int),
                "Bathrooms": best["Bathrooms"].astype(int),
                "Max area (sq ft)": best["Area"].astype(int),
                "Floors": best["Floors"].astype(int),
                "Garage": best["Garage"],
                "Estimated price": best["Price"].map(format_inr),
                "Likely range": [f"{inr_in_words(p * (1 - range_pct / 100))} – "
                                 f"{inr_in_words(p * (1 + range_pct / 100))}"
                                 for p in best["Price"]],
                "Budget used": (best["Price"] / budget_val * 100).round(0).astype(int).astype(str) + "%",
            })
            st.caption("Best (largest) house you can afford for each bedroom/bathroom layout:")
            st.dataframe(show, hide_index=True, width="stretch")

            csv_out = best.assign(**{
                "Price": best["Price"].round(),
                "Range low": (best["Price"] * (1 - range_pct / 100)).round(),
                "Range high": (best["Price"] * (1 + range_pct / 100)).round(),
            })[["Bedrooms", "Bathrooms", "Area", "Floors", "Garage", "YearBuilt", "Location",
                "Condition", "Price", "Range low", "Range high"]]
            st.download_button("⬇ Download results (CSV)", csv_out.to_csv(index=False).encode(),
                               file_name="houses_in_budget.csv", mime="text/csv")
            st.caption("Estimates use the sidebar assumptions. They are not official valuations.")