"""
Streamlit app for PlanLA Impact Simulator.
Interactive dashboard for visualizing Olympic investment impacts on LA neighborhoods.
"""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
from data_loader import load_la_data
from simulation import simulate_localized_investments, generate_summary_stats, mock_llm_summary
from network_effects import build_neighborhood_network
import geopandas as gpd

# Load environment variables from .env file
load_dotenv()

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import requests for Hugging Face
try:
    import requests
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="PlanLA Impact Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean professional styling
# Color palette: White (#FFFFFF), Medium Gray (#5A6C7D), Light Gray (#F5F5F5)
st.markdown("""
<style>
    /* Main app background - White */
    .stApp {
        background-color: #FFFFFF;
    }

    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #FFFFFF;
    }

    /* Sidebar styling - Light Gray */
    [data-testid="stSidebar"] {
        background-color: #F5F5F5;
    }

    [data-testid="stSidebar"] .element-container {
        color: #5A6C7D;
    }

    /* Headers - Medium Gray */
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #5A6C7D;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    .sub-header {
        font-size: 1.1rem;
        color: #5A6C7D;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
        opacity: 0.7;
    }

    /* All text - Medium Gray */
    .main, .stMarkdown, p, span, label {
        color: #5A6C7D !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #F5F5F5;
        border-radius: 0.375rem;
        color: #5A6C7D;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border: 1px solid #E0E0E0;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FF8C42;
        color: #FFFFFF !important;
        border-color: #FF8C42;
    }

    /* Headers visibility */
    h1, h2, h3, h4, h5, h6 {
        color: #5A6C7D !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Buttons - Light Gray */
    .stButton > button {
        background-color: #E8E8E8;
        color: #5A6C7D;
        border: none;
        border-radius: 0.375rem;
        font-weight: 500;
    }

    .stButton > button:hover {
        background-color: #D0D0D0;
    }

    /* Primary button - Orange highlight */
    .stButton > button[kind="primary"] {
        background-color: #FF8C42;
        color: #FFFFFF;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #FF7A29;
    }

    /* Download button */
    .stDownloadButton > button {
        background-color: #5A6C7D;
        color: #FFFFFF;
        border: none;
        border-radius: 0.375rem;
    }

    .stDownloadButton > button:hover {
        background-color: #4a5b6b;
    }

    /* Info/Success/Warning boxes - using light gray background with dark text */
    .stAlert {
        background-color: #F5F5F5 !important;
        color: #5A6C7D !important;
        border: 1px solid #E0E0E0;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #5A6C7D !important;
    }

    [data-testid="stMetricLabel"] {
        color: #5A6C7D !important;
    }

    /* Input fields */
    .stTextInput input, .stSelectbox select {
        border-color: #E0E0E0;
        color: #5A6C7D;
    }
</style>
""", unsafe_allow_html=True)

def create_choropleth_map(df_sim, use_real_data=False):
    """
    Create a choropleth map showing displacement risk by neighborhood.
    
    Args:
        df_sim (pd.DataFrame): Simulated neighborhood data
        use_real_data (bool): Whether using real LA data
    
    Returns:
        plotly.graph_objects.Figure: Choropleth map
    """
    if use_real_data and hasattr(df_sim, 'geometry'):
        # Use real GeoDataFrame with actual geometries
        return create_real_choropleth_map(df_sim)
    else:
        # Use mock data with point coordinates
        return create_mock_choropleth_map(df_sim)

def create_real_choropleth_map(gdf):
    """
    Create choropleth map using real GeoDataFrame geometries.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with real geometries

    Returns:
        plotly.graph_objects.Figure: Choropleth map
    """
    import json

    # Prepare data with proper index
    gdf_plot = gdf.copy()
    gdf_plot['id'] = gdf_plot.index.astype(str)

    # Convert GeoDataFrame to GeoJSON with proper feature IDs
    geojson = json.loads(gdf_plot.to_json())

    # Ensure each feature has an id property
    for i, feature in enumerate(geojson['features']):
        feature['id'] = str(i)

    fig = px.choropleth_mapbox(
        gdf_plot,
        geojson=geojson,
        locations='id',
        featureidkey='id',
        color='simulated_displacement_risk',
        hover_name='neighborhood',
        hover_data={
            'neighborhood': False,
            'simulated_displacement_risk': ':.1f',
            'rent_change_pct': ':.1f',
            'base_rent': ':.0f',
            'simulated_rent': ':.0f',
            'permit_density': ':.1f',
            'rent_burden_pct': ':.1f',
            'id': False
        },
        color_continuous_scale=[
            [0.0, '#2ECC71'],   # Green - Low risk (0-20%)
            [0.2, '#F1C40F'],   # Yellow - Low-Medium risk (20-40%)
            [0.4, '#E67E22'],   # Orange - Medium risk (40-60%)
            [0.6, '#E74C3C'],   # Red - Medium-High risk (60-80%)
            [1.0, '#8B0000']    # Dark Red - High risk (80-100%)
        ],
        range_color=[0, 100],
        labels={'simulated_displacement_risk': 'Displacement Risk (%)'},
        mapbox_style="open-street-map",
        center={"lat": 34.0522, "lon": -118.2437},
        zoom=9,
        title="Displacement Risk Heatmap by Neighborhood",
        opacity=0.6
    )

    fig.update_layout(
        height=600,
        margin={"r": 0, "t": 30, "l": 0, "b": 0}
    )

    return fig

def create_mock_choropleth_map(df_sim):
    """
    Create choropleth map using point data when no geometry is available.
    
    Args:
        df_sim (pd.DataFrame): Simulated neighborhood data
    
    Returns:
        plotly.graph_objects.Figure: Choropleth map
    """
    # Use lat/lon columns if available, otherwise use centroids
    if 'lat' in df_sim.columns and 'lon' in df_sim.columns:
        map_df = df_sim.copy()
    else:
        # Create simple point data based on neighborhood names
        map_data = []
        for _, row in df_sim.iterrows():
            # Simple fallback coordinates (would be better with real neighborhood boundaries)
            lat = 34.0522 + (hash(row['neighborhood']) % 100) / 1000  # Rough LA area
            lon = -118.2437 + (hash(row['neighborhood']) % 100) / 1000
            map_data.append({
                'neighborhood': row['neighborhood'],
                'lat': lat,
                'lon': lon,
                'simulated_displacement_risk': row['simulated_displacement_risk'],
                'rent_change_pct': row['rent_change_pct'],
                'base_rent': row['base_rent'],
                'simulated_rent': row['simulated_rent']
            })
        map_df = pd.DataFrame(map_data)
    
    # Create scatter map
    fig = px.scatter_mapbox(
        map_df,
        lat='lat',
        lon='lon',
        color='simulated_displacement_risk',
        size='simulated_rent',
        hover_data={
            'neighborhood': True,
            'simulated_displacement_risk': ':.1f',
            'rent_change_pct': ':.1f',
            'base_rent': ':.0f',
            'simulated_rent': ':.0f',
            'lat': False,
            'lon': False
        },
        color_continuous_scale=[
            [0.0, '#2ECC71'],   # Green - Low risk
            [0.2, '#F1C40F'],   # Yellow - Low-Medium risk
            [0.4, '#E67E22'],   # Orange - Medium risk
            [0.6, '#E74C3C'],   # Red - Medium-High risk
            [1.0, '#8B0000']    # Dark Red - High risk
        ],
        range_color=[0, 100],
        labels={'simulated_displacement_risk': 'Displacement Risk (%)'},
        size_max=20,
        zoom=9,
        center={"lat": 34.0522, "lon": -118.2437},  # Center on LA
        mapbox_style="open-street-map",
        title="Displacement Risk Heatmap by Neighborhood"
    )
    
    fig.update_layout(
        height=600,
        margin={"r": 0, "t": 30, "l": 0, "b": 0}
    )
    
    return fig

def create_rent_change_chart(df_sim):
    """
    Create a bar chart showing rent changes by neighborhood.
    
    Args:
        df_sim (pd.DataFrame): Simulated neighborhood data
    
    Returns:
        plotly.graph_objects.Figure: Bar chart
    """
    # Sort by rent change percentage
    df_sorted = df_sim.sort_values('rent_change_pct', ascending=True)
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        y=df_sorted['neighborhood'],
        x=df_sorted['rent_change_pct'],
        orientation='h',
        marker_color=['red' if x > 0 else 'green' for x in df_sorted['rent_change_pct']],
        text=[f"{x:.1f}%" for x in df_sorted['rent_change_pct']],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Rent Change: %{x:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Rent Change by Neighborhood",
        xaxis_title="Rent Change (%)",
        yaxis_title="Neighborhood",
        height=500,
        showlegend=False
    )
    
    return fig

def generate_ai_response(user_query, df_sim, stats, current_investments):
    """
    Generate AI response to urban planning queries.
    Uses OpenAI API if available, otherwise provides rule-based intelligent responses.
    """
    # Calculate total investments per type across all neighborhoods
    total_transit = sum(inv.get('transit', 0) for inv in current_investments.values())
    total_affordable = sum(inv.get('affordable', 0) for inv in current_investments.values())
    total_community = sum(inv.get('community', 0) for inv in current_investments.values())
    total_invested = total_transit + total_affordable + total_community
    neighborhoods_invested = len(current_investments)

    # Prepare context about current simulation
    context = f"""
Current Investment Levels:
- Total Invested: ${total_invested/1_000_000:.0f}M across {neighborhoods_invested} neighborhoods
- Transit Infrastructure: ${total_transit/1_000_000:.0f}M
- Affordable Housing: ${total_affordable/1_000_000:.0f}M
- Community Hubs: ${total_community/1_000_000:.0f}M

Neighborhoods with Investments:
"""
    for hood, inv in current_investments.items():
        hood_total = inv.get('transit', 0) + inv.get('affordable', 0) + inv.get('community', 0)
        context += f"\n- {hood}: ${hood_total/1_000_000:.0f}M"

    context += f"""

Current Results:
- Average rent change: ${stats['avg_rent_change']:.0f} ({stats['avg_rent_change_pct']:.1f}%)
- Average displacement risk change: {stats['avg_risk_change']:.1f} points
- High risk neighborhoods: {stats['high_risk_neighborhoods']}
- High rent increase neighborhoods: {stats['high_rent_increase_neighborhoods']}

Top 3 Most At-Risk Neighborhoods:
"""

    # Add top at-risk neighborhoods
    top_risk = df_sim.nlargest(3, 'simulated_displacement_risk')[['neighborhood', 'simulated_displacement_risk', 'rent_change_pct']]
    for _, row in top_risk.iterrows():
        context += f"\n- {row['neighborhood']}: {row['simulated_displacement_risk']:.1f}% risk, {row['rent_change_pct']:.1f}% rent increase"

    # Try Hugging Face first (free!)
    hf_api_key = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")

    if HF_AVAILABLE:
        try:
            # Use Hugging Face Inference API (free tier available)
            response = query_huggingface_model(user_query, context, hf_api_key)
            if response:
                return response
        except Exception as e:
            print(f"Hugging Face API error: {str(e)}")

    # Try OpenAI as fallback
    openai_key = os.getenv("OPENAI_API_KEY")
    if OPENAI_AVAILABLE and openai_key:
        try:
            client = OpenAI(api_key=openai_key)

            system_prompt = """You are an expert urban planning AI assistant helping city planners optimize Olympic infrastructure investments to minimize displacement while improving community outcomes.

You have access to simulation data showing how different investment types affect LA neighborhoods. Provide specific, actionable recommendations based on the data.

Key investment effects:
- Transit Infrastructure: Increases accessibility and rent (5-15%), slight displacement risk increase
- Community Hubs: Reduces displacement risk (up to 25%), minimal rent impact
- Affordable Housing: Strongly reduces displacement (up to 40%), slightly lowers rents
- Green Spaces: Increases desirability and rent (up to 8%), minor displacement risk
- Mixed-Use Development: Increases density and rent (up to 12%), can reduce displacement if done well

Always be specific about which neighborhoods need attention and suggest concrete investment combinations."""

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{context}\n\nUser Question: {user_query}"}
                ],
                temperature=0.7,
                max_tokens=500
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI API error: {str(e)}")

    # Rule-based intelligent responses (fallback)
    return generate_rule_based_response(user_query, df_sim, stats, current_investments)

def query_huggingface_model(user_query, context, api_key=None):
    """Query Hugging Face Inference API for LLM response using InferenceClient."""

    if not api_key:
        return None

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=api_key)

        system_prompt = """You are an expert urban planning AI assistant. Analyze the simulation data and provide specific, actionable recommendations for optimizing Olympic infrastructure investments to minimize displacement.

Key investment effects:
- Transit Infrastructure: +5-15% rent, slight displacement risk
- Community Hubs: -25% displacement risk, minimal rent impact
- Affordable Housing: -40% displacement risk, -5% rent
- Green Spaces: +8% rent, minor displacement risk
- Mixed-Use: +12% rent, -15% displacement if done well

Be concise and specific about neighborhoods and investment levels."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context}\n\nUser Question: {user_query}"}
        ]

        response = client.chat_completion(
            messages=messages,
            model="meta-llama/Llama-3.1-8B-Instruct",
            max_tokens=400,
            temperature=0.7
        )

        generated_text = response.choices[0].message.content.strip()

        # Format nicely
        if not generated_text.startswith('**'):
            generated_text = f"**AI Analysis:**\n\n{generated_text}"

        return generated_text

    except Exception as e:
        print(f"Hugging Face error: {str(e)}")
        # If model is loading or error, fall back
        if "503" in str(e) or "loading" in str(e).lower():
            st.info("AI model is loading... Using rule-based response for now.")
        return None

def generate_rule_based_response(user_query, df_sim, stats, current_investments):
    """Generate intelligent rule-based responses without AI API."""
    query_lower = user_query.lower()

    # Identify top at-risk neighborhoods
    top_risk_areas = df_sim.nlargest(3, 'simulated_displacement_risk')
    top_rent_increase = df_sim.nlargest(3, 'rent_change_pct')

    # Analyze query intent
    if any(word in query_lower for word in ['reduce', 'lower', 'minimize', 'decrease']) and 'displacement' in query_lower:
        response = f"**Strategy to Reduce Displacement:**\n\n"
        response += f"Based on current data, {stats['high_risk_neighborhoods']} neighborhoods have high displacement risk (>70%). "
        response += f"Here's my recommendation:\n\n"
        response += f"**Priority Actions:**\n"
        response += f"1. **Target Affordable Housing** in high-risk neighborhoods ($30M per neighborhood) - This has the strongest anti-displacement effect\n"
        response += f"2. **Add Community Hubs** ($10M per neighborhood) - Provides stability and social support\n\n"
        response += f"**Focus on these neighborhoods:**\n"
        for _, row in top_risk_areas.head(3).iterrows():
            response += f"- **{row['neighborhood']}**: {row['simulated_displacement_risk']:.1f}% risk, ${row['base_rent']:.0f} base rent\n"
        response += f"\n**Trade-off:** This will slightly reduce overall rents but dramatically improve community stability."

    elif 'transit' in query_lower and any(word in query_lower for word in ['balance', 'while', 'but']):
        response = f"**Balancing Transit with Anti-Displacement:**\n\n"
        response += f"Transit infrastructure is important for Olympic access, but it can drive gentrification. Here's a balanced approach:\n\n"
        response += f"**Recommended Mix:**\n"
        response += f"1. **Transit**: Keep at 40-50% (measured expansion)\n"
        response += f"2. **Affordable Housing**: Increase to 80% near transit corridors\n"
        response += f"3. **Community Hubs**: 50-60% in affected areas\n\n"
        response += f"**Focus Areas** (close to Olympic venues):\n"
        close_venues = df_sim.nsmallest(3, 'distance_to_olympic_site')
        for _, row in close_venues.iterrows():
            response += f"- **{row['neighborhood']}**: {row['distance_to_olympic_site']:.1f}km from venues\n"
        response += f"\nThis protects existing residents while improving connectivity."

    elif any(word in query_lower for word in ['optimize', 'best', 'optimal', 'recommend']):
        response = f"**Optimal Investment Strategy:**\n\n"
        response += f"Based on current simulation data:\n\n"
        response += f"**Current Status:**\n"
        response += f"- Avg rent change: ${stats['avg_rent_change']:.0f} ({stats['avg_rent_change_pct']:.1f}%)\n"
        response += f"- Risk change: {stats['avg_risk_change']:.1f} points\n"
        response += f"- High-risk areas: {stats['high_risk_neighborhoods']}\n\n"
        response += f"**Recommended Investment Levels:**\n"
        response += f"1. **Affordable Housing: 75%** - Strongest displacement prevention\n"
        response += f"2. **Community Hubs: 60%** - Community stability\n"
        response += f"3. **Transit: 45%** - Balanced accessibility\n"
        response += f"4. **Green Spaces: 40%** - Quality of life\n"
        response += f"5. **Mixed-Use: 30%** - Controlled density\n\n"
        response += f"**Expected Impact:** ~30% reduction in displacement risk with moderate rent increases."

    elif any(neighborhood in query_lower for neighborhood in df_sim['neighborhood'].str.lower().values):
        # Neighborhood-specific query
        mentioned_hood = None
        for hood in df_sim['neighborhood'].values:
            if hood.lower() in query_lower:
                mentioned_hood = hood
                break

        if mentioned_hood:
            hood_data = df_sim[df_sim['neighborhood'] == mentioned_hood].iloc[0]
            response = f"**Analysis for {mentioned_hood}:**\n\n"
            response += f"**Current Status:**\n"
            response += f"- Base Rent: ${hood_data['base_rent']:.0f}\n"
            response += f"- Simulated Rent: ${hood_data['simulated_rent']:.0f} ({hood_data['rent_change_pct']:.1f}% change)\n"
            response += f"- Displacement Risk: {hood_data['simulated_displacement_risk']:.1f}%\n"
            response += f"- Distance to Olympic Sites: {hood_data['distance_to_olympic_site']:.1f}km\n\n"

            if hood_data['simulated_displacement_risk'] > 70:
                response += f"**HIGH RISK AREA**\n\n**Priority Actions:**\n"
                response += f"1. Deploy 80% Affordable Housing immediately\n"
                response += f"2. Establish Community Hubs at 70%\n"
                response += f"3. Limit market-rate development\n"
            elif hood_data['simulated_displacement_risk'] > 50:
                response += f"**MODERATE RISK**\n\n**Recommended Actions:**\n"
                response += f"1. Affordable Housing: 60%\n"
                response += f"2. Community Hubs: 50%\n"
                response += f"3. Monitor rent trends closely\n"
            else:
                response += f"**LOWER RISK**\n\n**Opportunity for:**\n"
                response += f"1. Transit expansion (40-50%)\n"
                response += f"2. Mixed-use development (40%)\n"
                response += f"3. Green spaces (50%)\n"
        else:
            response = "I couldn't find that neighborhood in the dataset. Try asking about Downtown LA, Hollywood, Santa Monica, or other LA neighborhoods."

    else:
        response = f"**General Recommendations:**\n\n"
        response += f"I can help you with:\n"
        response += f"- Reducing displacement in specific neighborhoods\n"
        response += f"- Balancing transit development with anti-displacement measures\n"
        response += f"- Optimizing investment combinations\n"
        response += f"- Analyzing specific neighborhoods\n\n"
        response += f"**Quick Stats:**\n"
        response += f"- {stats['high_risk_neighborhoods']} neighborhoods at high risk\n"
        response += f"- Average rent change: ${stats['avg_rent_change']:.0f}\n\n"
        response += f"Try asking: *'How can I reduce displacement in Downtown LA?'* or *'What's the optimal investment strategy?'*"

    return response

def main():
    """Main Streamlit app function."""

    # Header with professional context and LA 2028 Olympics branding
    col1, col2 = st.columns([6, 1])

    with col1:
        st.markdown('<h1 class="main-header">PlanLA Impact Simulator</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">LA 2028 Olympics • Evidence-Based Infrastructure Investment Planning</p>', unsafe_allow_html=True)

    with col2:
        # LA 2028 Olympics logo - top right
        try:
            from PIL import Image
            logo = Image.open("olympics-image.png")
            st.image(logo, width=250)
        except:
            st.markdown("🏅")  # Fallback to emoji if image not found

    # Initialize session state for investments and network
    if 'investments' not in st.session_state:
        st.session_state.investments = {}
    if 'selected_neighborhood' not in st.session_state:
        st.session_state.selected_neighborhood = None
    if 'network' not in st.session_state:
        st.session_state.network = None
    if 'total_budget' not in st.session_state:
        st.session_state.total_budget = 500_000_000  # $500M default budget

    # Calculate current spending
    total_spent = sum(
        inv['transit'] + inv['affordable'] + inv['community']
        for inv in st.session_state.investments.values()
    )
    remaining_budget = st.session_state.total_budget - total_spent

    # Sidebar controls
    st.sidebar.header("Neighborhood Investments")

    # Budget display
    st.sidebar.metric(
        "Budget Remaining",
        f"${remaining_budget/1_000_000:.0f}M",
        f"of ${st.session_state.total_budget/1_000_000:.0f}M"
    )

    if remaining_budget < 0:
        st.sidebar.error("Over budget! Remove some investments.")
    elif remaining_budget < 50_000_000:
        st.sidebar.warning(f"Only ${remaining_budget/1_000_000:.0f}M remaining")

    st.sidebar.markdown("---")
    
    # Load real LA data
    with st.spinner("Loading LA data..."):
        try:
            df = load_la_data(use_cache=True, force_refresh=False)
            if hasattr(df, 'geometry'):
                # Convert GeoDataFrame to DataFrame for simulation
                df_for_sim = pd.DataFrame(df.drop(columns=['geometry']))
                data_source = "Real LA Data"
                has_geometry = True
            else:
                df_for_sim = df
                data_source = "Real LA Data (No Geometry)"
                has_geometry = False
        except Exception as e:
            st.error("Unable to load LA data")
            st.error(str(e))
            st.info("**Troubleshooting tips:**")
            st.info("""
            - Check your internet connection
            - The LA APIs may be temporarily unavailable
            - Try refreshing the page in a few minutes
            - Contact the administrator if the issue persists
            """)
            st.stop()

    # Build network once
    if st.session_state.network is None:
        with st.spinner("Building neighborhood network..."):
            st.session_state.network = build_neighborhood_network(df_for_sim, distance_threshold=5.0)

    # Neighborhood selector and investment sliders in sidebar
    st.sidebar.markdown("---")
    selected = st.sidebar.selectbox(
        "Select Neighborhood to Configure",
        options=['None'] + sorted(df_for_sim['neighborhood'].tolist()),
        index=0 if st.session_state.selected_neighborhood is None else
              sorted(df_for_sim['neighborhood'].tolist()).index(st.session_state.selected_neighborhood) + 1,
        key="neighborhood_selector"
    )

    if selected != 'None':
        st.session_state.selected_neighborhood = selected
    else:
        st.session_state.selected_neighborhood = None

    if st.session_state.selected_neighborhood:
        hood = st.session_state.selected_neighborhood
        st.sidebar.success(f"Configuring: **{hood}**")

        st.sidebar.markdown("---")
        st.sidebar.markdown("**Investment Amounts ($M)**")

        # Get current saved values or defaults
        current_investments = st.session_state.investments.get(hood, {'transit': 0, 'affordable': 0, 'community': 0})

        # Investment sliders (using temporary keys)
        transit_val = st.sidebar.slider(
            "Transit Infrastructure",
            min_value=0,
            max_value=200,
            value=current_investments['transit'] // 1_000_000,
            step=10,
            help="Metro lines, bus rapid transit ($M)",
            key=f"transit_slider_{hood}"
        )

        affordable_val = st.sidebar.slider(
            "Affordable Housing",
            min_value=0,
            max_value=200,
            step=10,
            value=current_investments['affordable'] // 1_000_000,
            help="Deed-restricted affordable units ($M)",
            key=f"affordable_slider_{hood}"
        )

        community_val = st.sidebar.slider(
            "Community Hubs",
            min_value=0,
            max_value=100,
            step=5,
            value=current_investments['community'] // 1_000_000,
            help="Community centers, social services ($M)",
            key=f"community_slider_{hood}"
        )

        # Show total for this configuration
        hood_total = (transit_val + affordable_val + community_val)

        st.sidebar.markdown("---")
        st.sidebar.metric("Total for this Neighborhood", f"${hood_total:.0f}M")

        # Apply button
        col1, col2 = st.sidebar.columns(2)

        with col1:
            if st.button("Apply Investment", key="apply_btn", use_container_width=True, type="primary"):
                # Calculate new total if this investment is applied
                new_investment = {
                    'transit': transit_val * 1_000_000,
                    'affordable': affordable_val * 1_000_000,
                    'community': community_val * 1_000_000
                }

                # Calculate what total would be with this change
                temp_investments = st.session_state.investments.copy()
                temp_investments[hood] = new_investment
                new_total = sum(
                    inv['transit'] + inv['affordable'] + inv['community']
                    for inv in temp_investments.values()
                )

                # Check budget
                if new_total > st.session_state.total_budget:
                    st.sidebar.error(f"This would exceed budget by ${(new_total - st.session_state.total_budget)/1_000_000:.0f}M")
                else:
                    # Save the investments
                    st.session_state.investments[hood] = new_investment
                    # Remove if all zero
                    if transit_val == 0 and affordable_val == 0 and community_val == 0:
                        if hood in st.session_state.investments:
                            del st.session_state.investments[hood]
                    st.rerun()

        with col2:
            # Remove neighborhood button
            if st.button("Remove", key="remove_btn", use_container_width=True):
                if hood in st.session_state.investments:
                    del st.session_state.investments[hood]
                st.session_state.selected_neighborhood = None
                st.rerun()

    else:
        st.sidebar.markdown("")  # Empty space

    # Clear all investments button
    if len(st.session_state.investments) > 0:
        st.sidebar.markdown("---")
        if st.sidebar.button("Clear All Investments", key="clear_all_btn", use_container_width=True):
            st.session_state.investments = {}
            st.session_state.selected_neighborhood = None
            st.rerun()

    # Run simulation with localized investments
    df_sim = simulate_localized_investments(
        df_for_sim,
        st.session_state.investments,
        st.session_state.network
    )

    # Merge geometry back in for mapping
    if has_geometry:
        df_sim['geometry'] = df['geometry'].values
        df_sim = gpd.GeoDataFrame(df_sim, geometry='geometry')

    stats = generate_summary_stats(df_sim)
    
    # Professional Impact Dashboard
    st.markdown("### Key Impact Indicators")

    col1, col2, col3 = st.columns(3)

    with col1:
        delta_color = "inverse" if stats['avg_rent_change'] > 0 else "normal"
        st.metric(
            "Average Rent Change",
            f"${stats['avg_rent_change']:.0f}/mo",
            f"{stats['avg_rent_change_pct']:.1f}%",
            delta_color=delta_color,
            help="Change in average monthly rent across all neighborhoods"
        )

    with col2:
        total_hoods = len(df_sim)
        pct_high_risk = (stats['high_risk_neighborhoods'] / total_hoods) * 100
        st.metric(
            "High-Risk Communities",
            f"{stats['high_risk_neighborhoods']} of {total_hoods}",
            f"{pct_high_risk:.0f}%",
            delta_color="inverse",
            help="Neighborhoods with >70% displacement risk"
        )

    with col3:
        pct_high_rent = (stats['high_rent_increase_neighborhoods'] / total_hoods) * 100
        st.metric(
            "Significant Rent Increases",
            f"{stats['high_rent_increase_neighborhoods']} of {total_hoods}",
            f"{pct_high_rent:.0f}%",
            delta_color="inverse",
            help="Neighborhoods experiencing >10% rent increases"
        )

    # Investment summary
    total_invested = sum(
        inv['transit'] + inv['affordable'] + inv['community']
        for inv in st.session_state.investments.values()
    )
    neighborhoods_invested = len(st.session_state.investments)

    with st.expander("Investment Summary", expanded=False):
        st.markdown(f"""
        **Investment Overview**:
        - Total Invested: ${total_invested/1_000_000:.0f}M
        - Neighborhoods with Investments: {neighborhoods_invested}

        **Invested Neighborhoods**:
        """)
        for hood, inv in st.session_state.investments.items():
            total_hood = inv['transit'] + inv['affordable'] + inv['community']
            st.markdown(f"- **{hood}**: ${total_hood/1_000_000:.0f}M")

        st.markdown(f"""

        **Projected Outcomes**:
        - Total neighborhoods analyzed: {len(df_sim)}
        - Average displacement risk: {df_sim['simulated_displacement_risk'].mean():.1f}%
        - Average monthly rent: ${df_sim['simulated_rent'].mean():.0f}
        - Neighborhoods protected from high risk: {total_hoods - stats['high_risk_neighborhoods']}
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Map View", "Data Table", "Charts", "Summary", "AI Assistant", "About"])
    
    with tab1:
        st.subheader("Displacement Risk Heatmap")

        map_fig = create_choropleth_map(df_sim, has_geometry)
        st.plotly_chart(map_fig, use_container_width=True)
    
    with tab2:
        st.subheader("Neighborhood Data")
        
        # Display options
        col1, col2 = st.columns([1, 1])
        with col1:
            available_columns = df_sim.columns.tolist()
            default_columns = ['neighborhood', 'base_rent', 'simulated_rent', 'rent_change_pct', 
                             'baseline_displacement_risk', 'simulated_displacement_risk', 'risk_change',
                             'permit_density', 'rent_burden_pct']
            
            show_columns = st.multiselect(
                "Select columns to display:",
                options=available_columns,
                default=default_columns
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort by:",
                options=['neighborhood', 'rent_change_pct', 'simulated_displacement_risk'],
                index=1
            )
        
        # Display table
        display_df = df_sim[show_columns].sort_values(sort_by, ascending=False)
        st.dataframe(
            display_df,
            width='stretch',
            hide_index=True
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"planla_simulation_{neighborhoods_invested}_hoods_{total_invested/1_000_000:.0f}M.csv",
            mime="text/csv"
        )
    
    with tab3:
        st.subheader("Rent Change Analysis")
        
        chart_fig = create_rent_change_chart(df_sim)
        st.plotly_chart(chart_fig, use_container_width=True)
        
        # Additional insights
        st.markdown("### Key Insights")
        
        highest_increase = df_sim.loc[df_sim['rent_change_pct'].idxmax()]
        highest_risk = df_sim.loc[df_sim['simulated_displacement_risk'].idxmax()]
        
        st.info(f"""
        **Highest Rent Increase**: {highest_increase['neighborhood']} 
        ({highest_increase['rent_change_pct']:.1f}% increase, ${highest_increase['rent_change']:.0f})
        
        **Highest Displacement Risk**: {highest_risk['neighborhood']} 
        ({highest_risk['simulated_displacement_risk']:.1f}% risk)
        """)
    
    with tab4:
        st.subheader("Impact Analysis Summary")

        # Generate simplified summary
        st.markdown(f"""
        ## Impact Analysis Summary

        ### Current Investment Status
        - **Total Invested**: ${total_invested/1_000_000:.0f}M across {neighborhoods_invested} neighborhoods
        - **Network Effects**: Investments propagate to neighbors within 5km

        ### Key Findings

        {'**Rent Changes:**' if stats['avg_rent_change'] > 0 else '**Rent Stability:**'}
        - Average rent: ${stats['avg_rent_change']:.0f} ({stats['avg_rent_change_pct']:.1f}%)
        - {stats['high_rent_increase_neighborhoods']} neighborhoods see >10% rent increases

        **Displacement Risk:**
        - Average risk change: {stats['avg_risk_change']:.1f} points
        - {stats['high_risk_neighborhoods']} neighborhoods face displacement risk >70%

        ### Recommendations
        {'Consider additional affordable housing investments in high-risk areas' if stats['high_risk_neighborhoods'] > 5 else 'Current displacement risk is manageable'}
        {'Implement rent stabilization measures in affected neighborhoods' if stats['high_rent_increase_neighborhoods'] > 3 else 'Rent increases are within acceptable ranges'}
        """)

    with tab5:
        st.subheader("AI Urban Planning Assistant")
        st.markdown("Ask the AI assistant to help you optimize investment strategies to minimize displacement and improve community outcomes.")

        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm your AI urban planning assistant. I can help you:\n\n" +
                 "• Analyze displacement risks in specific neighborhoods\n" +
                 "• Suggest optimal investment combinations\n" +
                 "• Compare different planning scenarios\n" +
                 "• Explain the trade-offs between different investments\n\n" +
                 "Try asking: *'How can I reduce displacement in Downtown LA while still building transit?'*"}
            ]

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask about urban planning strategies..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_ai_response(prompt, df_sim, stats, st.session_state.investments)
                    st.markdown(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

    with tab6:
        st.subheader("About This Simulation")

        st.markdown("""
        ## Overview

        The PlanLA Impact Simulator models how 2028 Olympic infrastructure investments affect Los Angeles neighborhoods,
        focusing on displacement risk and housing affordability. The simulation uses real data from 114 LA neighborhoods
        and incorporates network effects to show how investments propagate across geographically connected areas.

        ---

        ## Data Sources

        ### Real LA Data
        - **LA Times Neighborhood Boundaries**: 114 official LA neighborhoods with polygon geometries
        - **US Census Bureau**: American Community Survey (ACS) 5-Year 2022 rent burden data from 2,498 census tracts
        - **Spatial Aggregation**: Census tract data spatially joined to neighborhoods using geometric intersections

        ### Synthetic Data
        - **Building Permits**: Estimated permit density when API unavailable
        - **Base Rent**: Estimated from rent burden + median income when direct data unavailable
        - **Olympic Proximity**: Calculated using distance to LA Memorial Coliseum, Crypto.com Arena, and SoFi Stadium

        ---

        ## Displacement Risk Calculation

        Displacement risk is a composite score (0-100) calculated from three factors:

        ### 1. Rent Burden (Weight: 30%)
        ```
        rent_burden = (monthly_rent × 12) / median_household_income
        ```
        Higher rent-to-income ratios indicate greater housing cost stress.

        ### 2. Development Pressure (Weight: 20%)
        ```
        development_pressure = building_permit_density / 10
        ```
        More construction permits suggest increased gentrification pressure.

        ### 3. Olympic Proximity (Weight: 50%)
        ```
        proximity_factor = max(0, (50km - distance_to_venues) / 50km)
        ```
        Neighborhoods closer to Olympic venues face higher displacement risk due to speculation and development.

        ### Combined Formula
        ```python
        displacement_risk = (
            rent_burden × 30 +
            development_pressure × 20 +
            proximity_factor × 50
        )
        # Clipped to 0-100 range
        ```

        ---

        ## Network Effects Algorithm

        Investments propagate to nearby neighborhoods using spatial network analysis:

        ### 1. Network Construction
        ```python
        # Build weighted graph based on geographic proximity
        for each pair of neighborhoods:
            distance = haversine(neighborhood_A, neighborhood_B)
            if distance <= 5km:
                weight = 1 / (1 + distance)  # Inverse distance weighting
                add_edge(A, B, weight)
        ```

        ### 2. Spillover Calculation
        ```python
        # For each neighborhood receiving spillover
        spillover_effect = 0
        for neighbor in adjacent_neighborhoods:
            if neighbor has investment:
                distance_weight = edge_weight(neighbor, target)
                normalized_weight = distance_weight / sum(all_edge_weights)
                spillover_effect += (
                    neighbor_investment_effect ×
                    normalized_weight ×
                    0.3  # 30% spillover strength
                )
        ```

        ### 3. Distance Threshold
        - **Maximum propagation distance**: 5 kilometers
        - **Spillover strength**: 30% of direct effect
        - **Weighting**: Inverse distance (closer = stronger effect)

        ---

        ## Budget Constraints

        - **Total Budget**: $500 million (configurable)
        - **Enforcement**: Real-time validation prevents overspending
        - **Tracking**: Live budget remaining display in sidebar
        - **Warnings**: Alerts when budget is low or exceeded

        ---

        ## Technical Implementation

        ### Network Centrality (PageRank)
        Neighborhoods are ranked by connectivity influence using eigenvector centrality,
        identifying strategic investment locations that maximize spillover reach.

        ### Spatial Join Algorithm
        Census tract rent burden data is aggregated to neighborhoods using:
        ```python
        # Area-weighted averaging for overlapping geometries
        for each neighborhood:
            intersecting_tracts = spatial_join(neighborhood, census_tracts)
            weighted_rent_burden = sum(
                tract.rent_burden × tract.intersection_area
            ) / total_intersection_area
        ```

        ### Real-Time Simulation
        All 114 neighborhoods are re-simulated on every interaction, incorporating:
        - Direct investment effects
        - Network spillover propagation
        - Budget constraints
        - Displacement risk recalculation

        ---

        ## AI Urban Planning Assistant

        The simulator features an integrated AI assistant powered by advanced language models that analyzes your current investment strategy and provides context-aware recommendations.

        ### How It Works

        **1. Real-Time Context Awareness**
        - The AI has access to your current simulation state, including all neighborhood investments
        - Automatically analyzes displacement risk patterns, rent changes, and network effects
        - Identifies the top 3 most at-risk neighborhoods in your current scenario

        **2. Intelligent Response System**
        The assistant uses a three-tier approach:
        - **Hugging Face LLM** (Primary): Free, open-source Llama 3.1-8B model for natural language understanding
        - **OpenAI GPT-4** (Fallback): Premium model for complex urban planning queries (requires API key)
        - **Rule-Based Intelligence** (Always Available): Built-in expert system with pattern-matching for common planning scenarios

        **3. Context-Rich Analysis**
        For every query, the AI receives:
        ```
        - Total investments: $X million across Y neighborhoods
        - Investment breakdown: Transit, Affordable Housing, Community Hubs
        - Current simulation results: Average rent change, displacement risk levels
        - Neighborhood-specific data: Rent burden, proximity to Olympic venues
        - Network effects: Spillover impacts on neighboring areas
        ```

        ### Technical Details

        **Language Model Configuration**
        - Model: Meta Llama 3.1-8B-Instruct (Hugging Face)
        - Max tokens: 400 (concise, actionable responses)
        - Temperature: 0.7 (balanced creativity and consistency)
        - System prompt: Expert urban planning assistant specializing in Olympic infrastructure and displacement prevention

        **Rule-Based Fallback Intelligence**
        When AI models are unavailable, the assistant uses:
        - Pattern matching on user queries (displacement, transit, optimization, neighborhoods)
        - Statistical analysis of simulation data
        - Pre-programmed urban planning heuristics
        - Context-aware recommendations based on current investment levels

        ---

        ## Limitations & Assumptions

        1. **Simplified Causality**: Real-world displacement is more complex than modeled relationships
        2. **Static Network**: Geographic relationships don't change over time in this model
        3. **Linear Effects**: Investment impacts assumed to be linear and additive
        4. **No Time Dynamics**: Instantaneous effects rather than gradual changes over years
        5. **Synthetic Data**: Some variables estimated when real data unavailable
        6. **AI Limitations**: AI responses are based on simulation data and may not capture all real-world nuances
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        f"**PlanLA Impact Simulator** | Built with Streamlit | "
        f"Data: {data_source} | "
        f"Simulation: Olympic investment impact modeling"
    )

if __name__ == "__main__":
    main()
