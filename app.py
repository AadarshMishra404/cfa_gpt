import streamlit as st
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
def _get_secret(key, default=""):
    try:
        return st.secrets[key]
    except (FileNotFoundError, KeyError):
        return os.environ.get(key, default)

OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
FINE_TUNED_MODEL = _get_secret("FINE_TUNED_MODEL", "ft:gpt-4o-mini-2024-07-18:northstar:cfa-expert-v2:DIC778WZ")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STAR_RATINGS_FILE = "star_ratings_detailed_1.csv"
SCENARIOS_DIR = os.path.join(BASE_DIR, "scenarios")

# Original 10 scenarios (root directory)
ORIGINAL_SCENARIOS = {
    "Mutual Funds Portfolio": "scenario1_mutual_funds.json",
    "Equities/Stocks Portfolio": "scenario2_stocks_holdings.json",
    "Combined Portfolio (Deposit + MF + Equities)": "scenario3_combined_portfolio.json",
    "Fixed Deposits Portfolio": "scenario4_fixed_deposits.json",
    "Retirement NPS Portfolio": "scenario5_retirement_nps.json",
    "Gold & Government Bonds Portfolio": "scenario6_gold_bonds.json",
    "Young Professional (24, Single, Aggressive)": "scenario7_young_professional.json",
    "Mid-Career Family (35, Married, 2 Kids)": "scenario8_mid_career_family.json",
    "Pre-Retirement (55, Conservative)": "scenario9_pre_retirement.json",
    "Self-Employed Business Owner (40, Chennai)": "scenario10_business_owner.json",
}

@st.cache_data
def discover_scenarios():
    """Auto-discover all scenario JSON files from scenarios/ subfolders."""
    import glob
    categories = {"Original (1-10)": {k: os.path.join(BASE_DIR, v) for k, v in ORIGINAL_SCENARIOS.items()}}

    if not os.path.isdir(SCENARIOS_DIR):
        return categories

    for cat_folder in sorted(os.listdir(SCENARIOS_DIR)):
        cat_path = os.path.join(SCENARIOS_DIR, cat_folder)
        if not os.path.isdir(cat_path):
            continue
        cat_label = cat_folder.replace("_", " ").title()
        files = sorted(glob.glob(os.path.join(cat_path, "*.json")))
        if not files:
            continue
        cat_scenarios = {}
        for f in files:
            basename = os.path.basename(f).replace(".json", "")
            # Try to extract label from JSON
            try:
                with open(f, "r") as fh:
                    d = json.load(fh)
                demo = d.get("demographics", {})
                name = demo.get("name", "")
                age = demo.get("age", "")
                city = demo.get("city", "")
                occ = demo.get("occupation", {}).get("designation", "")
                label = f"{name}, {age}, {occ}, {city}" if name else basename
            except Exception:
                label = basename
            cat_scenarios[label] = f
        categories[cat_label] = cat_scenarios

    return categories


# ============================================================
# STAR RATINGS LOADER
# ============================================================
@st.cache_data
def load_star_ratings():
    """Load star ratings CSV into a dict keyed by ISIN."""
    ratings = {}
    csv_path = os.path.join(BASE_DIR, STAR_RATINGS_FILE)
    if not os.path.exists(csv_path):
        return ratings
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        isin = str(row.get("ISIN", "")).strip()
        if not isin:
            continue
        ratings[isin] = {
            "fund_name": row.get("FundName", ""),
            "category": row.get("Category", ""),
            "star_rating": int(row["StarRating"]) if pd.notna(row.get("StarRating")) else None,
            "star_3y": int(row["Star_3Y"]) if pd.notna(row.get("Star_3Y")) else None,
            "rank_3y": int(row["Rank_3Y"]) if pd.notna(row.get("Rank_3Y")) else None,
            "star_5y": int(float(row["Star_5Y"])) if pd.notna(row.get("Star_5Y")) else None,
            "rank_5y": int(float(row["Rank_5Y"])) if pd.notna(row.get("Rank_5Y")) else None,
            "star_10y": int(float(row["Star_10Y"])) if pd.notna(row.get("Star_10Y")) else None,
            "rank_10y": int(float(row["Rank_10Y"])) if pd.notna(row.get("Rank_10Y")) else None,
            "mrar_3y": round(float(row["MRAR2_3Y"]), 6) if pd.notna(row.get("MRAR2_3Y")) else None,
            "mrar_5y": round(float(row["MRAR2_5Y"]), 6) if pd.notna(row.get("MRAR2_5Y")) else None,
            "years_used": row.get("YearsUsed", ""),
        }
    return ratings


def get_star_info(isin, star_ratings):
    """Get star rating info for a single ISIN."""
    return star_ratings.get(isin, {})


def find_alternatives(isin, star_ratings, top_n=3):
    """Find top-rated alternatives in the same category."""
    current = star_ratings.get(isin, {})
    category = current.get("category", "")
    if not category:
        return []
    peers = []
    for peer_isin, info in star_ratings.items():
        if peer_isin == isin:
            continue
        if info["category"] != category:
            continue
        if info.get("star_rating") is not None and info["star_rating"] >= 4:
            peers.append({"isin": peer_isin, **info})
    peers.sort(key=lambda x: (x["star_rating"] or 0, x["mrar_3y"] or 0), reverse=True)
    return peers[:top_n]


STAR_RATINGS = load_star_ratings()

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CFA Portfolio Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1a73e8, #4fc3f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-card h3 {
        font-size: 0.85rem;
        opacity: 0.85;
        margin-bottom: 5px;
    }
    .metric-card h1 {
        font-size: 1.8rem;
        margin: 0;
    }
    .positive { color: #00c853; }
    .negative { color: #ff1744; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
    .investor-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-green { background: #e8f5e9; color: #2e7d32; }
    .badge-red { background: #ffebee; color: #c62828; }
    .badge-orange { background: #fff3e0; color: #e65100; }
    div[data-testid="stExpander"] details summary {
        font-size: 1.05rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA EXTRACTION FUNCTIONS (from cfa_analyzer.py)
# ============================================================
def load_scenario(file_path):
    """Load scenario JSON. Accepts full path or filename (for original 10)."""
    if os.path.isabs(file_path) and os.path.exists(file_path):
        path = file_path
    else:
        path = os.path.join(BASE_DIR, file_path)
    with open(path, "r") as f:
        return json.load(f)


def extract_investor_profile(data):
    """Extract investor profile from AA data + top-level demographics if present."""
    profile_out = {}

    # Try extracting from AA account holder data
    for item in data.get("data", []):
        detail = item.get("dataDetail", {})
        json_data = detail.get("jsonData", {})
        account = json_data.get("Account", {})
        profile = account.get("Profile", {})
        holders = profile.get("Holders", {})
        holder = holders.get("Holder", {})
        if holder and holder.get("name"):
            age = "N/A"
            if holder.get("dob"):
                try:
                    dob = datetime.strptime(holder["dob"], "%Y-%m-%d")
                    age = (datetime.now() - dob).days // 365
                except ValueError:
                    pass
            profile_out = {
                "name": holder.get("name", "N/A"),
                "dob": holder.get("dob", "N/A"),
                "age": age,
                "mobile": holder.get("mobile", "N/A"),
                "email": holder.get("email", "N/A"),
                "pan": holder.get("pan", "N/A"),
                "nominee": holder.get("nominee", "N/A"),
                "kyc": holder.get("ckycCompliance", "N/A"),
            }
            break

    # Override/enrich with top-level demographics block (scenarios 7-10)
    demo = data.get("demographics", {})
    if demo:
        if demo.get("name"):
            profile_out["name"] = demo["name"]
        if demo.get("dob"):
            profile_out["dob"] = demo["dob"]
            try:
                dob = datetime.strptime(demo["dob"], "%Y-%m-%d")
                profile_out["age"] = (datetime.now() - dob).days // 365
            except ValueError:
                pass
        profile_out["gender"] = demo.get("gender", "N/A")
        profile_out["marital_status"] = demo.get("marital_status", "N/A")
        profile_out["residential_status"] = demo.get("residential_status", "RESIDENT")
        profile_out["city"] = demo.get("city", "N/A")
        profile_out["dependents"] = demo.get("dependents", [])
        profile_out["occupation"] = demo.get("occupation", {})
        profile_out["education"] = demo.get("education", {})

    # Attach top-level financial blocks if present
    profile_out["income"] = data.get("income", {})
    profile_out["expenses"] = data.get("expenses", {})
    profile_out["liabilities"] = data.get("liabilities", {})
    profile_out["insurance"] = data.get("insurance", {})
    profile_out["goals"] = data.get("goals", [])
    profile_out["tax"] = data.get("tax", {})

    return profile_out


def extract_all_accounts(data):
    accounts = {
        "deposit": [], "mutual_fund": [], "equities": [],
        "fixed_deposit": [], "nps": [], "govt_securities": [],
        "epf": [], "ppf": [],
    }
    for item in data.get("data", []):
        fip = item.get("fipid", "Unknown")
        detail = item.get("dataDetail", {})
        json_data = detail.get("jsonData", {})
        account = json_data.get("Account", {})
        acc_type = account.get("type", "UNKNOWN")
        summary = account.get("Summary", {})
        txns = account.get("Transactions", {}).get("Transaction", [])

        if acc_type == "DEPOSIT":
            # Check if this is a term deposit (FD) or savings
            dep_type = summary.get("type", "")
            if dep_type in ("TERM_DEPOSIT", "NRE_TERM_DEPOSIT") or summary.get("accountType") == "FIXED":
                fds = summary.get("FixedDeposits", {}).get("FixedDeposit", [])
                fd_total = sum(float(fd.get("currentValue", 0)) for fd in fds)
                accounts["fixed_deposit"].append({
                    "fip": fip, "total_value": fd_total,
                    "account_sub_type": summary.get("accountType", "FIXED"),
                    "deposits": fds, "transactions": txns,
                })
            else:
                accounts["deposit"].append({
                    "fip": fip,
                    "balance": float(summary.get("currentBalance", 0)),
                    "type": summary.get("type", "N/A"),
                    "account_sub_type": summary.get("accountType", "REGULAR"),
                    "branch": summary.get("branch", "N/A"),
                    "ifsc": summary.get("ifscCode", "N/A"),
                    "status": summary.get("status", "N/A"),
                    "opening_date": summary.get("openingDate", "N/A"),
                    "transactions": txns,
                })

        elif acc_type == "MUTUAL_FUND":
            holdings = summary.get("Holdings", {}).get("Holding", [])
            accounts["mutual_fund"].append({
                "fip": fip,
                "current_value": float(summary.get("currentValue", 0)),
                "cost_value": float(summary.get("costValue", 0)),
                "total_schemes": summary.get("totalNumSchemes", 0),
                "holdings": holdings,
                "transactions": txns,
            })

        elif acc_type == "EQUITIES":
            holdings = summary.get("Holdings", {}).get("Holding", [])
            accounts["equities"].append({
                "fip": fip,
                "current_value": float(summary.get("currentValue", 0)),
                "investment_value": float(summary.get("investmentValue", 0)),
                "total_holdings": summary.get("totalHoldings", 0),
                "holdings": holdings,
                "transactions": txns,
            })

        elif acc_type == "NPS":
            holdings = summary.get("Holdings", {}).get("Holding", [])
            accounts["nps"].append({
                "fip": fip,
                "current_value": float(summary.get("currentValue", 0)),
                "total_contribution": float(summary.get("totalContribution", 0)),
                "gains": float(summary.get("totalGains", 0)),
                "scheme_preference": summary.get("schemePreference", "N/A"),
                "tax_benefits": summary.get("TaxBenefits", {}),
                "retirement_projection": summary.get("RetirementProjection", {}),
                "holdings": holdings,
                "transactions": txns,
            })

        elif acc_type == "GOVERNMENT_SECURITIES":
            holdings = summary.get("Holdings", {}).get("Holding", [])
            accounts["govt_securities"].append({
                "fip": fip,
                "current_value": float(summary.get("currentValue", 0)),
                "investment_value": float(summary.get("investmentValue", 0)),
                "holdings": holdings,
                "transactions": txns,
            })

        elif acc_type == "EPF":
            accounts["epf"].append({
                "fip": fip,
                "balance": float(summary.get("currentBalance", 0)),
                "interest_rate": summary.get("interestRate", "N/A"),
                "member_since": summary.get("memberSince", "N/A"),
                "summary": summary,
            })

        elif acc_type == "PPF":
            accounts["ppf"].append({
                "fip": fip,
                "balance": float(summary.get("currentBalance", 0)),
                "interest_rate": summary.get("interestRate", "N/A"),
                "maturity_date": summary.get("maturityDate", summary.get("newMaturityDate", "N/A")),
                "summary": summary,
            })

    return accounts


def compute_portfolio_totals(accounts):
    deposit_total = sum(a["balance"] for a in accounts["deposit"])
    mf_current = sum(a["current_value"] for a in accounts["mutual_fund"])
    mf_cost = sum(a["cost_value"] for a in accounts["mutual_fund"])
    eq_current = sum(a["current_value"] for a in accounts["equities"])
    eq_invested = sum(a["investment_value"] for a in accounts["equities"])
    fd_total = sum(a["total_value"] for a in accounts.get("fixed_deposit", []))
    nps_current = sum(a["current_value"] for a in accounts.get("nps", []))
    nps_invested = sum(a["total_contribution"] for a in accounts.get("nps", []))
    gs_current = sum(a["current_value"] for a in accounts.get("govt_securities", []))
    gs_invested = sum(a["investment_value"] for a in accounts.get("govt_securities", []))
    epf_total = sum(a["balance"] for a in accounts.get("epf", []))
    ppf_total = sum(a["balance"] for a in accounts.get("ppf", []))

    total_value = deposit_total + mf_current + eq_current + fd_total + nps_current + gs_current + epf_total + ppf_total
    total_invested = deposit_total + mf_cost + eq_invested + fd_total + nps_invested + gs_invested + epf_total + ppf_total
    total_pnl = total_value - total_invested
    pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

    return {
        "deposit": deposit_total,
        "mf_current": mf_current,
        "mf_cost": mf_cost,
        "eq_current": eq_current,
        "eq_invested": eq_invested,
        "fd_total": fd_total,
        "nps_current": nps_current,
        "nps_invested": nps_invested,
        "gs_current": gs_current,
        "gs_invested": gs_invested,
        "epf_total": epf_total,
        "ppf_total": ppf_total,
        "total_value": total_value,
        "total_invested": total_invested,
        "total_pnl": total_pnl,
        "pnl_pct": pnl_pct,
    }


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================
def render_metric_card(label, value, delta=None, prefix=""):
    delta_html = ""
    if delta is not None:
        color = "positive" if delta >= 0 else "negative"
        sign = "+" if delta >= 0 else ""
        delta_html = f'<p class="{color}" style="font-size:1rem;margin:0;">{sign}{delta:.2f}%</p>'
    st.markdown(f"""
    <div class="metric-card">
        <h3>{label}</h3>
        <h1>{prefix}{value}</h1>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def format_inr(amount):
    """Format number in Indian currency style."""
    if amount >= 10000000:
        return f"{amount/10000000:.2f} Cr"
    elif amount >= 100000:
        return f"{amount/100000:.2f} L"
    else:
        return f"{amount:,.2f}"


def plot_asset_allocation(totals):
    labels, values, colors = [], [], []
    if totals["deposit"] > 0:
        labels.append("Bank Deposit")
        values.append(totals["deposit"])
        colors.append("#4fc3f7")
    if totals["mf_current"] > 0:
        labels.append("Mutual Funds")
        values.append(totals["mf_current"])
        colors.append("#7c4dff")
    if totals["eq_current"] > 0:
        labels.append("Equities")
        values.append(totals["eq_current"])
        colors.append("#00e676")

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+percent",
        textfont_size=13,
        hovertemplate="<b>%{label}</b><br>Value: ₹%{value:,.0f}<br>Share: %{percent}<extra></extra>",
    )])
    fig.update_layout(
        title=dict(text="Asset Allocation", font=dict(size=16)),
        showlegend=True,
        height=380,
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return fig


def plot_mf_holdings(holdings_list):
    rows = []
    for h in holdings_list:
        rows.append({
            "Scheme": h.get("schemeName", "")[:35],
            "Category": h.get("subCategory", h.get("category", "N/A")),
            "Current Value": float(h.get("currentValue", 0)),
            "Cost Value": float(h.get("costValue", 0)),
            "P&L": float(h.get("currentValue", 0)) - float(h.get("costValue", 0)),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Cost Value", x=df["Scheme"], y=df["Cost Value"],
        marker_color="#90a4ae", text=df["Cost Value"].apply(lambda x: f"₹{x:,.0f}"),
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Current Value", x=df["Scheme"], y=df["Current Value"],
        marker_color="#7c4dff", text=df["Current Value"].apply(lambda x: f"₹{x:,.0f}"),
        textposition="outside",
    ))
    fig.update_layout(
        title="Mutual Funds: Cost vs Current Value",
        barmode="group", height=400,
        margin=dict(t=50, b=80, l=40, r=20),
        yaxis_title="Value (INR)",
        xaxis_tickangle=-20,
    )
    return fig


def plot_mf_category_pie(holdings_list):
    rows = []
    for h in holdings_list:
        cat = h.get("subCategory", h.get("category", "OTHER"))
        rows.append({"Category": cat, "Value": float(h.get("currentValue", 0))})
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    df = df.groupby("Category", as_index=False).sum()
    fig = px.pie(df, names="Category", values="Value", hole=0.45,
                 color_discrete_sequence=px.colors.qualitative.Set2,
                 title="MF Category Allocation")
    fig.update_layout(height=350, margin=dict(t=50, b=20, l=20, r=20))
    return fig


def plot_eq_holdings(holdings_list):
    rows = []
    for h in holdings_list:
        rows.append({
            "Symbol": h.get("symbol", ""),
            "Holding Value": float(h.get("holdingValue", 0)),
            "P&L": float(h.get("pnl", 0)),
            "P&L %": float(h.get("pnlPercentage", 0)),
            "Sector": h.get("sector", "N/A").replace("_", " ").title(),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return None, None

    colors = ["#00c853" if v >= 0 else "#ff1744" for v in df["P&L"]]
    fig_pnl = go.Figure(go.Bar(
        x=df["Symbol"], y=df["P&L"],
        marker_color=colors,
        text=df["P&L"].apply(lambda x: f"₹{x:,.0f}"),
        textposition="outside",
    ))
    fig_pnl.update_layout(
        title="Stock-wise Profit & Loss",
        height=400, yaxis_title="P&L (INR)",
        margin=dict(t=50, b=40, l=40, r=20),
    )

    sector_df = df.groupby("Sector", as_index=False)["Holding Value"].sum()
    fig_sector = px.pie(sector_df, names="Sector", values="Holding Value", hole=0.45,
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        title="Sector Allocation")
    fig_sector.update_layout(height=350, margin=dict(t=50, b=20, l=20, r=20))
    return fig_pnl, fig_sector


def plot_txn_flow(transactions, acc_type):
    rows = []
    for t in transactions:
        t_type = t.get("type", "")
        amount = float(t.get("amount", 0))

        if acc_type == "DEPOSIT":
            date_str = t.get("transactionTimestamp", t.get("valueDate", ""))
            direction = "Inflow" if t_type == "CREDIT" else "Outflow"
        elif acc_type == "MUTUAL_FUND":
            date_str = t.get("orderDate", t.get("navDate", ""))
            direction = "Inflow" if t_type in ("BUY", "DIVIDEND_PAYOUT") else "Outflow"
        else:
            date_str = t.get("tradeDate", "")
            direction = "Inflow" if t_type in ("BUY", "DIVIDEND", "CORPORATE_ACTION") else "Outflow"

        if date_str:
            try:
                dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
                rows.append({"Date": dt, "Amount": amount, "Direction": direction, "Type": t_type})
            except ValueError:
                pass

    df = pd.DataFrame(rows)
    if df.empty:
        return None

    df = df.sort_values("Date")
    fig = px.bar(df, x="Date", y="Amount", color="Direction",
                 color_discrete_map={"Inflow": "#00c853", "Outflow": "#ff1744"},
                 hover_data=["Type"],
                 title="Transaction Flow (Last 12 Months)")
    fig.update_layout(height=350, margin=dict(t=50, b=40, l=40, r=20), xaxis_title="", yaxis_title="Amount (INR)")
    return fig


def plot_eq_holdings_treemap(holdings_list):
    rows = []
    for h in holdings_list:
        rows.append({
            "Symbol": h.get("symbol", ""),
            "Sector": h.get("sector", "OTHER").replace("_", " ").title(),
            "Value": float(h.get("holdingValue", 0)),
            "P&L %": float(h.get("pnlPercentage", 0)),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    fig = px.treemap(df, path=["Sector", "Symbol"], values="Value", color="P&L %",
                     color_continuous_scale="RdYlGn", title="Holdings Treemap (Size = Value, Color = P&L %)")
    fig.update_layout(height=420, margin=dict(t=50, b=20, l=20, r=20))
    return fig


# ============================================================
# CFA ANALYSIS (GPT CALL)
# ============================================================
def build_cfa_prompt(investor_profile, accounts, scenario_label):
    holdings_summary = []
    txn_summary = []

    # --- Bank Deposits ---
    for dep in accounts["deposit"]:
        holdings_summary.append({
            "account_type": "Bank Deposit", "fip": dep["fip"], "balance": dep["balance"],
            "sub_type": dep.get("account_sub_type", "SAVINGS"),
        })
        txn_summary.append({
            "account_type": "DEPOSIT", "total_transactions": len(dep["transactions"]),
            "total_inflow": sum(float(t.get("amount", 0)) for t in dep["transactions"] if t.get("type") == "CREDIT"),
            "total_outflow": sum(float(t.get("amount", 0)) for t in dep["transactions"] if t.get("type") == "DEBIT"),
        })

    # --- Fixed Deposits ---
    for fd_acc in accounts.get("fixed_deposit", []):
        fd_h = []
        for fd in fd_acc.get("deposits", []):
            fd_h.append({
                "fd_number": fd.get("fdNumber"), "principal": fd.get("principalAmount"),
                "current_value": fd.get("currentValue"), "rate": fd.get("interestRate"),
                "tenure_months": fd.get("tenureMonths"), "maturity_date": fd.get("maturityDate"),
                "payout_mode": fd.get("interestPayoutMode"), "tax_saver": fd.get("taxSaverFD", False),
                "status": fd.get("status"),
            })
        holdings_summary.append({
            "account_type": "Fixed Deposits", "total_value": fd_acc["total_value"], "deposits": fd_h,
        })
        txn_summary.append({
            "account_type": "FIXED_DEPOSIT", "total_transactions": len(fd_acc.get("transactions", [])),
        })

    # --- Mutual Funds ---
    for mf in accounts["mutual_fund"]:
        mf_h = []
        for h in mf["holdings"]:
            mf_h.append({
                "scheme": h.get("schemeName"), "category": h.get("category"),
                "sub_category": h.get("subCategory"), "cost_value": h.get("costValue"),
                "current_value": h.get("currentValue"), "risk_rating": h.get("riskRating"),
                "sip_active": h.get("sipActive"), "sip_amount": h.get("sipAmount"),
                "plan_type": h.get("planType", "DIRECT"),
            })
        holdings_summary.append({
            "account_type": "Mutual Funds", "total_current_value": mf["current_value"],
            "total_cost_value": mf["cost_value"], "holdings": mf_h,
        })
        txn_summary.append({
            "account_type": "MUTUAL_FUND", "total_transactions": len(mf["transactions"]),
        })

    # --- Equities ---
    for eq in accounts["equities"]:
        eq_h = []
        for h in eq["holdings"]:
            eq_h.append({
                "company": h.get("companyName"), "symbol": h.get("symbol"),
                "quantity": h.get("quantity"), "avg_cost": h.get("avgCostPrice"),
                "current_price": h.get("currentPrice"), "pnl": h.get("pnl"),
                "pnl_pct": h.get("pnlPercentage"), "sector": h.get("sector"),
            })
        holdings_summary.append({
            "account_type": "Equities", "total_current_value": eq["current_value"],
            "total_investment_value": eq["investment_value"], "holdings": eq_h,
        })
        txn_summary.append({
            "account_type": "EQUITIES", "total_transactions": len(eq["transactions"]),
        })

    # --- NPS ---
    for nps in accounts.get("nps", []):
        nps_h = []
        for h in nps.get("holdings", []):
            nps_h.append({
                "scheme": h.get("schemeName"), "scheme_type": h.get("schemeType"),
                "allocation_pct": h.get("allocationPercentage"), "current_value": h.get("currentValue"),
                "returns_pct": h.get("returnsPercentage"), "asset_class": h.get("assetClass"),
            })
        holdings_summary.append({
            "account_type": "NPS", "current_value": nps["current_value"],
            "total_contribution": nps["total_contribution"], "gains": nps["gains"],
            "scheme_preference": nps["scheme_preference"],
            "retirement_projection": nps.get("retirement_projection", {}),
            "holdings": nps_h,
        })
        txn_summary.append({
            "account_type": "NPS", "total_transactions": len(nps.get("transactions", [])),
        })

    # --- Government Securities / SGBs ---
    for gs in accounts.get("govt_securities", []):
        gs_h = []
        for h in gs.get("holdings", []):
            gs_h.append({
                "instrument": h.get("instrumentName"), "type": h.get("instrumentType"),
                "quantity": h.get("quantity"), "holding_value": h.get("holdingValue"),
                "investment_value": h.get("investmentValue"), "pnl": h.get("pnl"),
                "coupon_rate": h.get("couponRate"), "maturity_date": h.get("maturityDate"),
                "ytm": h.get("yieldToMaturity"),
            })
        holdings_summary.append({
            "account_type": "Government Securities & Gold Bonds",
            "current_value": gs["current_value"],
            "investment_value": gs["investment_value"], "holdings": gs_h,
        })
        txn_summary.append({
            "account_type": "GOVT_SECURITIES", "total_transactions": len(gs.get("transactions", [])),
        })

    # --- EPF ---
    for epf in accounts.get("epf", []):
        holdings_summary.append({
            "account_type": "EPF (Employee Provident Fund)",
            "balance": epf["balance"], "interest_rate": epf.get("interest_rate"),
        })

    # --- PPF ---
    for ppf in accounts.get("ppf", []):
        holdings_summary.append({
            "account_type": "PPF (Public Provident Fund)",
            "balance": ppf["balance"], "interest_rate": ppf.get("interest_rate"),
            "maturity_date": ppf.get("maturity_date"),
        })

    age = investor_profile.get("age", "Unknown")

    # Build star ratings context for MF holdings
    star_context_lines = []
    for part in holdings_summary:
        if part.get("account_type") == "Mutual Funds":
            for h in part.get("holdings", []):
                isin = h.get("isin", "")
                star_info = get_star_info(isin, STAR_RATINGS)
                sr = star_info.get("star_rating")
                if sr is not None:
                    line = (
                        f"- {h.get('scheme', 'N/A')}: "
                        f"Overall {sr}/5 stars, "
                        f"3Y Rating: {star_info.get('star_3y', 'N/A')}/5 "
                        f"(Rank {star_info.get('rank_3y', 'N/A')}), "
                        f"5Y Rating: {star_info.get('star_5y', 'N/A')}/5 "
                        f"(Rank {star_info.get('rank_5y', 'N/A')}), "
                        f"MRAR 3Y: {star_info.get('mrar_3y', 'N/A')}"
                    )
                    star_context_lines.append(line)
                    if sr <= 3:
                        alternatives = find_alternatives(isin, STAR_RATINGS, top_n=3)
                        if alternatives:
                            star_context_lines.append("  Better alternatives in same category:")
                            for a in alternatives:
                                star_context_lines.append(
                                    f"    - {a['fund_name']} ({a['star_rating']}/5, "
                                    f"MRAR 3Y: {a.get('mrar_3y', 'N/A')})"
                                )

    star_section = ""
    if star_context_lines:
        star_section = (
            "\n## Mutual Fund Star Ratings (Morningstar-style)\n"
            + "\n".join(star_context_lines) + "\n"
        )

    # Build demographics context (scenarios 7-10)
    demo_section = ""
    occupation = investor_profile.get("occupation", {})
    income = investor_profile.get("income", {})
    expenses = investor_profile.get("expenses", {})
    liabilities = investor_profile.get("liabilities", {})
    insurance = investor_profile.get("insurance", {})
    goals = investor_profile.get("goals", [])
    tax = investor_profile.get("tax", {})
    if occupation or income or goals:
        demo_lines = []
        if investor_profile.get("marital_status"):
            demo_lines.append(f"- Marital Status: {investor_profile.get('marital_status')}")
        if investor_profile.get("residential_status") and investor_profile.get("residential_status") != "RESIDENT":
            demo_lines.append(f"- Residential Status: {investor_profile.get('residential_status')}")
        if investor_profile.get("city"):
            demo_lines.append(f"- City: {investor_profile.get('city')}")
        deps = investor_profile.get("dependents", [])
        if deps:
            dep_str = ", ".join(f"{d.get('name','?')} ({d.get('relation','?')}, age {d.get('age','?')})" for d in deps)
            demo_lines.append(f"- Dependents: {dep_str}")
        if occupation:
            demo_lines.append(f"- Occupation: {occupation.get('designation', '')} at {occupation.get('employer', '')} ({occupation.get('type', '')})")
            if occupation.get("years_to_retirement"):
                demo_lines.append(f"- Years to Retirement: {occupation['years_to_retirement']}")
        if demo_lines:
            demo_section += "\n## Demographics\n" + "\n".join(demo_lines) + "\n"

    income_section = ""
    if income:
        income_section = f"\n## Income\n{json.dumps(income, indent=2)}\n"

    expense_section = ""
    if expenses:
        expense_section = f"\n## Expenses\n{json.dumps(expenses, indent=2)}\n"

    liability_section = ""
    if liabilities and any(v for k, v in liabilities.items() if v and k not in ("total_emi", "total_outstanding", "debt_to_income_ratio", "emi_to_income_ratio")):
        liability_section = f"\n## Liabilities\n{json.dumps(liabilities, indent=2)}\n"

    insurance_section = ""
    if insurance:
        insurance_section = f"\n## Insurance\n{json.dumps(insurance, indent=2)}\n"

    goals_section = ""
    if goals:
        goals_section = f"\n## Financial Goals\n{json.dumps(goals, indent=2)}\n"

    tax_section = ""
    if tax:
        tax_section = f"\n## Tax Profile\n{json.dumps(tax, indent=2)}\n"

    system_prompt = (
        "You are a CFA Level III charterholder and SEBI-registered investment advisor. "
        "Analyze the investor's COMPLETE financial picture using CFA Institute standards, "
        "modern portfolio theory, and Indian market context.\n\n"
        "KNOWLEDGE BASE:\n"
        "- Asset Allocation: age-based equity rule (100 - age = equity %), core-satellite approach\n"
        "- Fixed Deposits: compare rates vs inflation (6-7%), FD vs debt MF tax efficiency, TDS implications\n"
        "- NPS: max 75% equity in active choice, 60% lumpsum tax-free at retirement, 40% mandatory annuity, 80CCD(1B) extra 50K deduction\n"
        "- EPF/PPF: tax-free returns, PPF 15-year lock-in, EPF taxable if withdrawn before 5 years\n"
        "- SGBs: 2.5% coupon + gold appreciation, LTCG tax-free at maturity, 5-year early exit window\n"
        "- G-Secs: duration risk, YTM analysis, interest rate sensitivity\n"
        "- Insurance: term life = 10-15x annual income, health = min 10L metro / 5L non-metro\n"
        "- Self-employed: advance tax quarterly, business vs personal expense separation, keyman insurance, working capital management\n"
        "- ULIPs: high charges, compare with term + MF combo, surrender if lock-in complete\n"
        "- Tax: ELSS 80C up to 1.5L, LTCG >1.25L at 12.5%, STCG at 20%, 80CCD(1B) NPS 50K, 80D health premiums\n"
        "- Regular vs Direct plans: 0.5-1.5% expense ratio difference, always recommend direct\n"
        "- Emergency fund: 6 months expenses in liquid instruments\n"
        "- Star ratings (1-5): 4-5 RETAIN, 3 WATCH, 1-2 SWITCH\n\n"
        "FORMAT RULES:\n"
        "- Use bullet points, not paragraphs\n"
        "- Use tables only for allocation breakdowns and rebalancing actions\n"
        "- Each bullet: 1 line with key insight + number\n"
        "- Assume the reader is financially literate\n"
        "- Total response under 1200 words\n"
        "- Lead with the verdict, then data"
    )

    user_prompt = f"""
## Scenario: {scenario_label}

## Investor Profile
- Name: {investor_profile.get('name', 'N/A')}
- Age: {age} years
- DOB: {investor_profile.get('dob', 'N/A')}
- PAN: {investor_profile.get('pan', 'N/A')}
- Nominee Registered: {investor_profile.get('nominee', 'N/A')}
- KYC Status: {investor_profile.get('kyc', 'N/A')}
{demo_section}
## Portfolio Holdings
{json.dumps(holdings_summary, indent=2)}
{star_section}{income_section}{expense_section}{liability_section}{insurance_section}{goals_section}{tax_section}
## Transaction Activity (Last 12 Months)
{json.dumps(txn_summary, indent=2)}

---

Analyze the COMPLETE financial picture. If demographics/income/goals/liabilities/insurance data is provided, use it for deeper analysis. If only portfolio data is available, analyze based on holdings alone.

### 1. PORTFOLIO ANALYSIS
- Asset allocation table (equity / debt / cash / gold / NPS / EPF-PPF %)
- Diversification verdict (sector, market-cap, geography spread)
- Concentration risks (any single holding >15%? any sector >25%?)
- P&L summary across all asset classes
- Cost efficiency (direct vs regular plans, FD rates vs inflation)
- SIP discipline and savings rate assessment

### 2. INVESTMENT SUITABILITY & LIFE STAGE
- Age {age}: current vs recommended allocation — verdict
- Risk capacity based on income stability, dependents, liabilities
- Emergency fund: current vs needed (6x monthly expenses)
- Goal gap analysis: for each goal, current progress vs required (with SIP amounts if applicable)
- Retirement readiness: projected corpus vs target (if applicable)
- Insurance adequacy: life cover vs 10-15x income, health cover gaps
- Compliance: KYC, nominee status

### 3. RECOMMENDATIONS (prioritized)
Rebalancing table (Current % -> Target %) then:
- URGENT: emergency fund, insurance gaps, nominee registration
- Fund switches (star ratings, regular-to-direct, underperformers)
- SIP changes (specific amounts tied to goals)
- Debt management (prepay high-rate loans? EMI optimization)
- Tax optimization (unused 80C/80CCD/80D, LTCG harvesting, regime choice)
- NPS/EPF/PPF actions (if applicable)
- Self-employed specific: business vs personal separation, working capital, advance tax planning

### 4. RISK ASSESSMENT
- Portfolio risk metrics (beta estimate, drawdown in 20% crash)
- Concentration risks (name specific stocks/sectors/AMCs)
- Liquidity risk (locked assets: FD, NPS, PPF, ELSS vs liquid)
- Liability risk (debt-to-income, EMI burden)
- Behavioral flags from transactions (panic selling, tip-based buying, etc.)
- **Overall Risk Rating: Conservative / Moderate / Aggressive** (bold)
"""
    return system_prompt, user_prompt


def get_cfa_analysis(investor_profile, accounts, scenario_label):
    client = OpenAI(api_key=OPENAI_API_KEY)
    system_prompt, user_prompt = build_cfa_prompt(investor_profile, accounts, scenario_label)

    response = client.chat.completions.create(
        model=FINE_TUNED_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=3000,
    )
    return response.choices[0].message.content, response.usage.total_tokens


# ============================================================
# SIDEBAR
# ============================================================
all_categories = discover_scenarios()

with st.sidebar:
    st.markdown('<p class="main-header">CFA Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Fine-Tuned CFA Model</p>', unsafe_allow_html=True)
    st.divider()

    # Category dropdown
    cat_names = list(all_categories.keys())
    selected_cat = st.selectbox("Category", cat_names, index=0)

    # Scenario dropdown within category
    cat_scenarios = all_categories[selected_cat]
    scenario_names = list(cat_scenarios.keys())
    selected_scenario = st.selectbox("Scenario", scenario_names, index=0)

    st.divider()
    st.caption(f"{sum(len(v) for v in all_categories.values())} scenarios across {len(all_categories)} categories")

    st.divider()
    st.markdown("**Model Info**")
    st.code(FINE_TUNED_MODEL, language=None)

    st.divider()
    st.markdown("**Data Source**")
    st.caption("Account Aggregator (AA) Framework")
    st.caption("ReBIT FI Schema v2.0.0")
    st.caption(f"Star Ratings: {len(STAR_RATINGS):,} funds loaded")


# ============================================================
# MAIN CONTENT
# ============================================================
data = load_scenario(cat_scenarios[selected_scenario])
profile = extract_investor_profile(data)
accounts = extract_all_accounts(data)
totals = compute_portfolio_totals(accounts)

# Header
st.markdown(f'<p class="main-header">Portfolio Analysis Dashboard</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">{selected_scenario}</p>', unsafe_allow_html=True)

# ---- Investor Profile Bar ----
with st.container():
    cols = st.columns([2, 1, 1, 1, 1, 1])
    cols[0].markdown(f"**{profile.get('name', 'N/A')}** &nbsp; | &nbsp; Age: **{profile.get('age', 'N/A')}**")
    cols[1].markdown(f"PAN: `{profile.get('pan', 'N/A')}`")

    kyc = profile.get("kyc", "N/A")
    kyc_badge = "badge-green" if kyc == "YES" else "badge-red"
    cols[2].markdown(f'KYC: <span class="investor-badge {kyc_badge}">{kyc}</span>', unsafe_allow_html=True)

    nominee = profile.get("nominee", "N/A")
    nom_badge = "badge-green" if nominee == "REGISTERED" else "badge-red"
    cols[3].markdown(f'Nominee: <span class="investor-badge {nom_badge}">{nominee}</span>', unsafe_allow_html=True)

st.divider()

# ---- Top Metrics ----
m1, m2, m3, m4 = st.columns(4)
with m1:
    render_metric_card("Total Portfolio Value", f"₹{format_inr(totals['total_value'])}")
with m2:
    render_metric_card("Total Invested", f"₹{format_inr(totals['total_invested'])}")
with m3:
    render_metric_card("Total P&L", f"₹{format_inr(abs(totals['total_pnl']))}", delta=totals["pnl_pct"])
with m4:
    num_holdings = sum(len(a["holdings"]) for a in accounts["mutual_fund"]) + \
                   sum(len(a["holdings"]) for a in accounts["equities"])
    render_metric_card("Total Holdings", str(num_holdings))

st.markdown("<br>", unsafe_allow_html=True)

# ---- TABS ----
tab_overview, tab_mf, tab_eq, tab_txn, tab_cfa = st.tabs([
    "Overview", "Mutual Funds", "Equities", "Transactions", "CFA Analysis"
])

# =========== OVERVIEW TAB ===========
with tab_overview:
    col_pie, col_summary = st.columns([1, 1])

    with col_pie:
        fig_alloc = plot_asset_allocation(totals)
        st.plotly_chart(fig_alloc, use_container_width=True)

    with col_summary:
        st.markdown("#### Portfolio Breakdown")
        breakdown_data = []
        if totals["deposit"] > 0:
            breakdown_data.append({
                "Asset Class": "Bank Deposit",
                "Value (INR)": f"₹{format_inr(totals['deposit'])}",
                "% of Portfolio": f"{totals['deposit']/totals['total_value']*100:.1f}%",
                "P&L": "N/A",
            })
        if totals["mf_current"] > 0:
            mf_pnl = totals["mf_current"] - totals["mf_cost"]
            breakdown_data.append({
                "Asset Class": "Mutual Funds",
                "Value (INR)": f"₹{format_inr(totals['mf_current'])}",
                "% of Portfolio": f"{totals['mf_current']/totals['total_value']*100:.1f}%",
                "P&L": f"₹{format_inr(mf_pnl)} ({mf_pnl/totals['mf_cost']*100:.1f}%)" if totals['mf_cost'] > 0 else "N/A",
            })
        if totals["eq_current"] > 0:
            eq_pnl = totals["eq_current"] - totals["eq_invested"]
            breakdown_data.append({
                "Asset Class": "Equities",
                "Value (INR)": f"₹{format_inr(totals['eq_current'])}",
                "% of Portfolio": f"{totals['eq_current']/totals['total_value']*100:.1f}%",
                "P&L": f"₹{format_inr(eq_pnl)} ({eq_pnl/totals['eq_invested']*100:.1f}%)" if totals['eq_invested'] > 0 else "N/A",
            })

        if breakdown_data:
            st.dataframe(pd.DataFrame(breakdown_data), hide_index=True, use_container_width=True)

        # Quick risk flags
        st.markdown("#### Quick Flags")
        flags = []
        if profile.get("nominee") not in ("REGISTERED",):
            flags.append(("Nominee NOT registered", "badge-red"))
        if totals["deposit"] > 0 and totals["deposit"] < 100000:
            flags.append(("Emergency fund may be low", "badge-orange"))
        if totals["eq_current"] > 0 and totals["eq_current"] / totals["total_value"] > 0.7:
            flags.append(("High equity concentration (>70%)", "badge-orange"))
        if totals["mf_current"] == 0 and totals["eq_current"] == 0:
            flags.append(("No investment holdings found", "badge-red"))

        if not flags:
            st.markdown('<span class="investor-badge badge-green">No critical flags</span>', unsafe_allow_html=True)
        for msg, badge in flags:
            st.markdown(f'<span class="investor-badge {badge}">{msg}</span>&nbsp;', unsafe_allow_html=True)

# =========== MUTUAL FUNDS TAB ===========
with tab_mf:
    all_mf_holdings = []
    for mf in accounts["mutual_fund"]:
        all_mf_holdings.extend(mf["holdings"])

    if not all_mf_holdings:
        st.info("No mutual fund holdings in this scenario.")
    else:
        col1, col2 = st.columns([1.3, 1])
        with col1:
            fig = plot_mf_holdings(all_mf_holdings)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig_cat = plot_mf_category_pie(all_mf_holdings)
            if fig_cat:
                st.plotly_chart(fig_cat, use_container_width=True)

        # ---- Fund Quality Rating Chart ----
        st.markdown("#### Fund Quality Ratings")
        star_rows = []
        for h in all_mf_holdings:
            isin = h.get("isin", "")
            star_info = get_star_info(isin, STAR_RATINGS)
            scheme_short = h.get("schemeName", "N/A")[:40]
            sr = star_info.get("star_rating")
            if sr is not None:
                star_rows.append({
                    "Scheme": scheme_short,
                    "Overall": sr,
                    "3Y Rating": star_info.get("star_3y"),
                    "5Y Rating": star_info.get("star_5y"),
                    "MRAR 3Y": star_info.get("mrar_3y"),
                })

        if star_rows:
            star_df = pd.DataFrame(star_rows)
            colors_map = {1: "#ff1744", 2: "#ff6d00", 3: "#ffd600", 4: "#76ff03", 5: "#00e676"}
            bar_colors = [colors_map.get(r, "#90a4ae") for r in star_df["Overall"]]

            fig_stars = go.Figure(go.Bar(
                x=star_df["Overall"],
                y=star_df["Scheme"],
                orientation="h",
                marker_color=bar_colors,
                text=star_df["Overall"].apply(lambda x: "★" * x),
                textposition="inside",
                textfont=dict(size=14),
            ))
            fig_stars.update_layout(
                title="Fund Star Ratings (1-5)",
                xaxis=dict(title="Star Rating", range=[0, 5.5], dtick=1),
                height=50 + len(star_rows) * 60,
                margin=dict(t=50, b=40, l=200, r=20),
            )
            st.plotly_chart(fig_stars, use_container_width=True)
        else:
            st.caption("Star rating data not available for holdings in this scenario.")

        # ---- Holdings Table (enriched with stars) ----
        st.markdown("#### Holdings Detail")
        mf_rows = []
        for h in all_mf_holdings:
            cost = float(h.get("costValue", 0))
            curr = float(h.get("currentValue", 0))
            pnl = curr - cost
            isin = h.get("isin", "")
            star_info = get_star_info(isin, STAR_RATINGS)
            sr = star_info.get("star_rating")
            star_display = ("★" * sr + "☆" * (5 - sr)) if sr is not None else "N/A"

            mf_rows.append({
                "Scheme": h.get("schemeName", ""),
                "AMC": h.get("amc", ""),
                "Category": h.get("subCategory", h.get("category", "")),
                "Rating": star_display,
                "3Y Rank": star_info.get("rank_3y", "N/A"),
                "5Y Rank": star_info.get("rank_5y", "N/A"),
                "Units": h.get("units", ""),
                "NAV": f"₹{h.get('nav', '')}",
                "Cost": f"₹{cost:,.0f}",
                "Current": f"₹{curr:,.0f}",
                "P&L": f"₹{pnl:,.0f}",
                "P&L %": f"{pnl/cost*100:.1f}%" if cost > 0 else "N/A",
                "Risk": h.get("riskRating", ""),
                "SIP Active": "Yes" if h.get("sipActive") else "No",
                "SIP Amt": f"₹{h.get('sipAmount', 'N/A')}" if h.get("sipAmount") else "-",
            })
        st.dataframe(pd.DataFrame(mf_rows), hide_index=True, use_container_width=True)

        # ---- Alternative Fund Recommendations ----
        st.markdown("#### Alternative Fund Recommendations")
        st.caption("Showing top-rated alternatives for funds rated 3 stars or below")
        has_alternatives = False
        for h in all_mf_holdings:
            isin = h.get("isin", "")
            star_info = get_star_info(isin, STAR_RATINGS)
            sr = star_info.get("star_rating")
            if sr is not None and sr <= 3:
                has_alternatives = True
                scheme_name = h.get("schemeName", "Unknown")
                star_display = "★" * sr + "☆" * (5 - sr)
                cat_display = star_info.get("category", "").replace("_", " ")

                with st.expander(f"{scheme_name} — {star_display} ({sr}/5) | Category: {cat_display}", expanded=True):
                    alternatives = find_alternatives(isin, STAR_RATINGS, top_n=5)
                    if alternatives:
                        alt_rows = []
                        for a in alternatives:
                            a_stars = a.get("star_rating", 0)
                            alt_rows.append({
                                "Fund Name": a["fund_name"].replace("_", " "),
                                "Rating": "★" * a_stars + "☆" * (5 - a_stars),
                                "3Y Rating": f"{a.get('star_3y', 'N/A')}/5",
                                "5Y Rating": f"{a.get('star_5y', 'N/A')}/5",
                                "MRAR 3Y": f"{a.get('mrar_3y', 'N/A')}",
                                "ISIN": a["isin"],
                            })
                        st.dataframe(pd.DataFrame(alt_rows), hide_index=True, use_container_width=True)
                    else:
                        st.caption("No higher-rated alternatives found in the same category.")

        if not has_alternatives:
            st.success("All holdings are rated 4 stars or above. No switches recommended.")

# =========== EQUITIES TAB ===========
with tab_eq:
    all_eq_holdings = []
    for eq in accounts["equities"]:
        all_eq_holdings.extend(eq["holdings"])

    if not all_eq_holdings:
        st.info("No equity holdings in this scenario.")
    else:
        col1, col2 = st.columns([1.3, 1])
        with col1:
            fig_pnl, fig_sector = plot_eq_holdings(all_eq_holdings)
            if fig_pnl:
                st.plotly_chart(fig_pnl, use_container_width=True)
        with col2:
            if fig_sector:
                st.plotly_chart(fig_sector, use_container_width=True)

        # Treemap
        fig_tree = plot_eq_holdings_treemap(all_eq_holdings)
        if fig_tree:
            st.plotly_chart(fig_tree, use_container_width=True)

        # Holdings table
        st.markdown("#### Holdings Detail")
        eq_rows = []
        for h in all_eq_holdings:
            eq_rows.append({
                "Symbol": h.get("symbol", ""),
                "Company": h.get("companyName", ""),
                "Qty": h.get("quantity", ""),
                "Avg Cost": f"₹{h.get('avgCostPrice', '')}",
                "CMP": f"₹{h.get('currentPrice', '')}",
                "Value": f"₹{float(h.get('holdingValue', 0)):,.0f}",
                "P&L": f"₹{float(h.get('pnl', 0)):,.0f}",
                "P&L %": f"{h.get('pnlPercentage', '')}%",
                "Sector": h.get("sector", "").replace("_", " ").title(),
                "Exchange": h.get("exchange", ""),
            })
        st.dataframe(pd.DataFrame(eq_rows), hide_index=True, use_container_width=True)

# =========== TRANSACTIONS TAB ===========
with tab_txn:
    for dep in accounts["deposit"]:
        if dep["transactions"]:
            with st.expander(f"Bank Transactions ({dep['fip']})", expanded=True):
                fig = plot_txn_flow(dep["transactions"], "DEPOSIT")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                credit = sum(float(t.get("amount", 0)) for t in dep["transactions"] if t.get("type") == "CREDIT")
                debit = sum(float(t.get("amount", 0)) for t in dep["transactions"] if t.get("type") == "DEBIT")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Transactions", len(dep["transactions"]))
                c2.metric("Total Inflow", f"₹{format_inr(credit)}")
                c3.metric("Total Outflow", f"₹{format_inr(debit)}")

    for mf in accounts["mutual_fund"]:
        if mf["transactions"]:
            with st.expander(f"Mutual Fund Transactions ({mf['fip']})", expanded=True):
                fig = plot_txn_flow(mf["transactions"], "MUTUAL_FUND")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                buys = sum(float(t.get("amount", 0)) for t in mf["transactions"] if t.get("type") == "BUY")
                sells = sum(float(t.get("amount", 0)) for t in mf["transactions"] if t.get("type") == "SELL")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Transactions", len(mf["transactions"]))
                c2.metric("Total Bought", f"₹{format_inr(buys)}")
                c3.metric("Total Sold", f"₹{format_inr(sells)}")

    for eq in accounts["equities"]:
        if eq["transactions"]:
            with st.expander(f"Equity Transactions ({eq['fip']})", expanded=True):
                fig = plot_txn_flow(eq["transactions"], "EQUITIES")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                buys = sum(float(t.get("amount", 0)) for t in eq["transactions"] if t.get("type") == "BUY")
                sells = sum(float(t.get("amount", 0)) for t in eq["transactions"] if t.get("type") == "SELL")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Transactions", len(eq["transactions"]))
                c2.metric("Total Bought", f"₹{format_inr(buys)}")
                c3.metric("Total Sold", f"₹{format_inr(sells)}")

    if not any(dep["transactions"] for dep in accounts["deposit"]) and \
       not any(mf["transactions"] for mf in accounts["mutual_fund"]) and \
       not any(eq["transactions"] for eq in accounts["equities"]):
        st.info("No transactions found in this scenario.")

# =========== CFA ANALYSIS TAB ===========
with tab_cfa:
    st.markdown("### CFA-Level Portfolio Analysis")
    st.caption(f"Powered by fine-tuned model: `{FINE_TUNED_MODEL}`")

    # Check for cached results
    cache_key = f"cfa_analysis_{selected_scenario}"

    if cache_key in st.session_state:
        st.markdown(st.session_state[cache_key]["analysis"])
        st.divider()
        st.caption(f"Tokens used: {st.session_state[cache_key]['tokens']}")
    else:
        st.markdown("""
        Click below to generate a comprehensive CFA analysis covering:
        - **Portfolio Analysis** — allocation, diversification, returns, costs
        - **Investment Suitability** — age-based, risk alignment, goals
        - **Recommendations** — rebalancing, tax optimization, SIP changes
        - **Risk Assessment** — market, concentration, liquidity, behavioral
        """)

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        run_analysis = st.button(
            "Generate CFA Analysis" if cache_key not in st.session_state else "Re-run Analysis",
            type="primary",
            use_container_width=True,
        )

    if run_analysis:
        with st.spinner("Analyzing portfolio with CFA fine-tuned model..."):
            try:
                analysis_text, tokens = get_cfa_analysis(profile, accounts, selected_scenario)
                st.session_state[cache_key] = {"analysis": analysis_text, "tokens": tokens}
                st.rerun()
            except Exception as e:
                st.error(f"Error calling model: {e}")
