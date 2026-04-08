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
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
FINE_TUNED_MODEL = st.secrets.get("FINE_TUNED_MODEL", os.environ.get("FINE_TUNED_MODEL", "ft:gpt-4o-mini-2024-07-18:northstar:cfa-expert-v2:DIC778WZ"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCENARIOS = {
    "Mutual Funds Portfolio": "scenario1_mutual_funds.json",
    "Equities/Stocks Portfolio": "scenario2_stocks_holdings.json",
    "Combined Portfolio (Deposit + MF + Equities)": "scenario3_combined_portfolio.json",
    "Fixed Deposits Portfolio": "scenario4_fixed_deposits.json",
    "Retirement NPS Portfolio": "scenario5_retirement_nps.json",
    "Gold & Government Bonds Portfolio": "scenario6_gold_bonds.json",
}
STAR_RATINGS_FILE = "star_ratings_detailed_1.csv"


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
def load_scenario(file_name):
    path = os.path.join(BASE_DIR, file_name)
    with open(path, "r") as f:
        return json.load(f)


def extract_investor_profile(data):
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
            return {
                "name": holder.get("name", "N/A"),
                "dob": holder.get("dob", "N/A"),
                "age": age,
                "mobile": holder.get("mobile", "N/A"),
                "email": holder.get("email", "N/A"),
                "pan": holder.get("pan", "N/A"),
                "nominee": holder.get("nominee", "N/A"),
                "kyc": holder.get("ckycCompliance", "N/A"),
            }
    return {}


def extract_all_accounts(data):
    accounts = {"deposit": [], "mutual_fund": [], "equities": []}
    for item in data.get("data", []):
        fip = item.get("fipid", "Unknown")
        detail = item.get("dataDetail", {})
        json_data = detail.get("jsonData", {})
        account = json_data.get("Account", {})
        acc_type = account.get("type", "UNKNOWN")
        summary = account.get("Summary", {})
        txns = account.get("Transactions", {}).get("Transaction", [])

        if acc_type == "DEPOSIT":
            accounts["deposit"].append({
                "fip": fip,
                "balance": float(summary.get("currentBalance", 0)),
                "type": summary.get("type", "N/A"),
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

    return accounts


def compute_portfolio_totals(accounts):
    deposit_total = sum(a["balance"] for a in accounts["deposit"])
    mf_current = sum(a["current_value"] for a in accounts["mutual_fund"])
    mf_cost = sum(a["cost_value"] for a in accounts["mutual_fund"])
    eq_current = sum(a["current_value"] for a in accounts["equities"])
    eq_invested = sum(a["investment_value"] for a in accounts["equities"])

    total_value = deposit_total + mf_current + eq_current
    total_invested = deposit_total + mf_cost + eq_invested
    total_pnl = total_value - total_invested
    pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

    return {
        "deposit": deposit_total,
        "mf_current": mf_current,
        "mf_cost": mf_cost,
        "eq_current": eq_current,
        "eq_invested": eq_invested,
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

    for dep in accounts["deposit"]:
        holdings_summary.append({"account_type": "Bank Deposit", "fip": dep["fip"], "balance": dep["balance"]})
        txn_summary.append({
            "account_type": "DEPOSIT", "total_transactions": len(dep["transactions"]),
            "total_inflow": sum(float(t.get("amount", 0)) for t in dep["transactions"] if t.get("type") == "CREDIT"),
            "total_outflow": sum(float(t.get("amount", 0)) for t in dep["transactions"] if t.get("type") == "DEBIT"),
        })

    for mf in accounts["mutual_fund"]:
        mf_h = []
        for h in mf["holdings"]:
            mf_h.append({
                "scheme": h.get("schemeName"), "category": h.get("category"),
                "sub_category": h.get("subCategory"), "cost_value": h.get("costValue"),
                "current_value": h.get("currentValue"), "risk_rating": h.get("riskRating"),
                "sip_active": h.get("sipActive"), "sip_amount": h.get("sipAmount"),
            })
        holdings_summary.append({
            "account_type": "Mutual Funds", "total_current_value": mf["current_value"],
            "total_cost_value": mf["cost_value"], "holdings": mf_h,
        })
        txn_summary.append({
            "account_type": "MUTUAL_FUND", "total_transactions": len(mf["transactions"]),
        })

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

    system_prompt = (
        "You are a CFA-certified financial advisor. Analyze the following investor portfolio "
        "data obtained via Account Aggregator (AA) framework. Provide a CONCISE but complete analysis "
        "structured into exactly four sections. Base your reasoning on CFA Institute standards, "
        "modern portfolio theory, and Indian market context.\n\n"
        "FORMAT RULES:\n"
        "- Use bullet points, not paragraphs\n"
        "- Use tables only for allocation breakdowns and rebalancing actions\n"
        "- Each bullet should be 1 line with the key insight + number\n"
        "- No explanations of what terms mean — assume the reader is financially literate\n"
        "- Total response should be under 800 words\n"
        "- Lead with the verdict, then the data supporting it"
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

## Portfolio Holdings
{json.dumps(holdings_summary, indent=2)}
{star_section}
## Transaction Activity (Last 12 Months)
{json.dumps(txn_summary, indent=2)}

---

Use the star ratings and MRAR data to evaluate fund quality. If rated 3 stars or below, recommend the suggested alternatives. If 4-5 stars, recommend retaining.

Keep the response CONCISE — use bullet points, not paragraphs. Under 800 words total. Lead with verdict, then supporting data.

### 1. PORTFOLIO ANALYSIS
Cover in bullet points with numbers:
- Asset allocation table (equity/debt/cash %)
- Diversification verdict (sector + market-cap spread)
- Top concentration risks (any holding >15%?)
- P&L summary (total invested vs current, overall return %)
- Cost efficiency (direct plans? expense ratio estimate)
- SIP discipline (monthly amount, consistency)

### 2. INVESTMENT SUITABILITY
Cover in bullet points:
- Age {age}: current vs recommended allocation — is it suitable? (one line verdict)
- Risk capacity assessment (one line)
- Liquidity gap: emergency fund current vs needed
- Goal alignment: wealth creation / retirement / short-term readiness
- Compliance flags: KYC and nominee status

### 3. RECOMMENDATIONS
Use a rebalancing table (Current % -> Target %) then bullet points for:
- Fund switches (based on star ratings — name specific funds)
- SIP changes (specific amounts)
- Tax actions (ELSS, LTCG harvesting — with numbers)
- Emergency fund action (how much to move where)
- Critical gaps (insurance, nominee — if applicable)

### 4. RISK ASSESSMENT
Cover in bullet points:
- Portfolio beta and expected drawdown in a 20% market crash
- Biggest concentration risk (name the stock/sector/AMC)
- Liquidity risk verdict (one line)
- Behavioral flags from transaction patterns
- **Overall Risk Rating: Conservative / Moderate / Aggressive** (bold, one line)
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
        max_tokens=2000,
    )
    return response.choices[0].message.content, response.usage.total_tokens


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<p class="main-header">CFA Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Fine-Tuned CFA Model</p>', unsafe_allow_html=True)
    st.divider()

    selected_scenario = st.selectbox("Select Scenario", list(SCENARIOS.keys()), index=2)
    st.divider()

    st.markdown("**Model Info**")
    st.code(FINE_TUNED_MODEL, language=None)

    st.divider()
    st.markdown("**Data Source**")
    st.caption("Account Aggregator (AA) Framework")
    st.caption("ReBIT FI Schema v2.0.0")
    st.caption(f"Star Ratings: {len(STAR_RATINGS):,} funds loaded")

    st.divider()
    st.markdown("**Scenarios**")
    st.caption("1 - Mutual Funds Only")
    st.caption("2 - Equities/Stocks Only")
    st.caption("3 - Combined Portfolio")


# ============================================================
# MAIN CONTENT
# ============================================================
data = load_scenario(SCENARIOS[selected_scenario])
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
