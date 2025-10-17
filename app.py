"""
Streamlit app for PlanLA Impact Simulator.
Interactive dashboard for visualizing Olympic investment impacts on LA neighborhoods.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
from data_loader import load_la_data
from simulation import simulate_olympic_impacts, generate_summary_stats, mock_llm_summary

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
        background-color: #5A6C7D;
        color: #FFFFFF !important;
        border-color: #5A6C7D;
    }

    /* Headers visibility */
    h1, h2, h3, h4, h5, h6 {
        color: #5A6C7D !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Buttons - Lighter Medium Gray */
    .stButton > button {
        background-color: #5A6C7D;
        color: #FFFFFF;
        border: none;
        border-radius: 0.375rem;
    }

    .stButton > button:hover {
        background-color: #4a5b6b;
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
    # Convert GeoDataFrame to GeoJSON for Plotly
    gdf_json = gdf.to_json()
    
    fig = px.choropleth_mapbox(
        gdf,
        geojson=gdf_json,
        locations=gdf.index,
        color='simulated_displacement_risk',
        hover_data={
            'neighborhood': True,
            'simulated_displacement_risk': ':.1f',
            'rent_change_pct': ':.1f',
            'base_rent': ':.0f',
            'simulated_rent': ':.0f',
            'permit_density': ':.1f',
            'rent_burden_pct': ':.1f'
        },
        color_continuous_scale='Reds',
        mapbox_style="open-street-map",
        center={"lat": 34.0522, "lon": -118.2437},
        zoom=9,
        title="Displacement Risk by Neighborhood (Real LA Data)"
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
        color_continuous_scale='Reds',
        size_max=20,
        zoom=9,
        center={"lat": 34.0522, "lon": -118.2437},  # Center on LA
        mapbox_style="open-street-map",
        title="Displacement Risk by Neighborhood (Point Data)"
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
    # Prepare context about current simulation
    context = f"""
Current Investment Levels:
- Transit Infrastructure: {current_investments['transit']}%
- Community Hubs: {current_investments['community_hubs']}%
- Affordable Housing: {current_investments['affordable_housing']}%
- Green Spaces: {current_investments['green_spaces']}%
- Mixed-Use Development: {current_investments['mixed_use']}%

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
        response += f"1. **Increase Affordable Housing** to 70-80% (currently {current_investments['affordable_housing']}%) - This has the strongest anti-displacement effect (up to 40% reduction)\n"
        response += f"2. **Boost Community Hubs** to 60% (currently {current_investments['community_hubs']}%) - Provides stability and social support\n\n"
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

    # Header with professional context
    st.markdown('<h1 class="main-header">PlanLA Impact Simulator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evidence-Based Olympic Infrastructure Investment Planning</p>', unsafe_allow_html=True)

    # Context/Mission Box
    with st.expander("About This Tool", expanded=False):
        st.markdown("""
        ### Purpose
        Model the impact of 2028 Olympic infrastructure investments on LA neighborhoods, focusing on:
        - **Displacement Risk**: Likelihood of existing residents being forced to relocate
        - **Rent Changes**: Impact on housing affordability
        - **Network Effects**: How investments in one area affect surrounding neighborhoods

        ### How It Works
        1. **Adjust investment levels** using the controls below
        2. **View impacts** across 15 LA neighborhoods with real data
        3. **Analyze network effects** - see how investments spread through connected areas
        4. **Compare scenarios** to find optimal allocation strategies

        ### Key Metrics
        - **Displacement Risk Index** (0-100): Composite score based on rent burden, development pressure, and Olympic proximity
        - **Network Centrality**: How connected/influential a neighborhood is in the urban network
        - **Spillover Effects**: Impact propagation to adjacent neighborhoods
        """)

    # Sidebar controls
    st.sidebar.header("Investment Scenario Builder")

    st.sidebar.markdown("**Investment Strategy:**")
    st.sidebar.caption("Adjust intensity levels to model different policy scenarios")

    # Add scenario presets
    scenario_preset = st.sidebar.selectbox(
        "Load Scenario Preset",
        ["Custom", "Transit-Focused", "Equity-Focused", "Balanced", "No Investment (Baseline)"],
        help="Quick-load common planning scenarios"
    )

    # Set values based on preset
    if scenario_preset == "Transit-Focused":
        transit_default, affordable_default, community_default = 80, 20, 30
        green_default, mixed_default = 40, 60
    elif scenario_preset == "Equity-Focused":
        transit_default, affordable_default, community_default = 30, 90, 80
        green_default, mixed_default = 20, 30
    elif scenario_preset == "Balanced":
        transit_default, affordable_default, community_default = 50, 50, 50
        green_default, mixed_default = 50, 50
    elif scenario_preset == "No Investment (Baseline)":
        transit_default = affordable_default = community_default = 0
        green_default = mixed_default = 0
    else:  # Custom
        transit_default = affordable_default = community_default = 0
        green_default = mixed_default = 0

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Investment Intensity (% of Max):**")

    transit_investment = st.sidebar.slider(
        "Transit Infrastructure",
        min_value=0,
        max_value=100,
        value=transit_default,
        step=5,
        help="Transit lines, stations, bus rapid transit. Higher values = more extensive network coverage."
    )

    affordable_housing = st.sidebar.slider(
        "Affordable Housing",
        min_value=0,
        max_value=100,
        value=affordable_default,
        step=5,
        help="Deed-restricted affordable units. Strongest anti-displacement intervention."
    )

    community_hub_investment = st.sidebar.slider(
        "Community Hubs & Services",
        min_value=0,
        max_value=100,
        value=community_default,
        step=5,
        help="Community centers, job training, social services. Builds social cohesion and resilience."
    )

    green_spaces = st.sidebar.slider(
        "Green Spaces & Parks",
        min_value=0,
        max_value=100,
        value=green_default,
        step=5,
        help="Public parks and green infrastructure. Improves quality of life but may increase property values."
    )

    mixed_use_development = st.sidebar.slider(
        "Mixed-Use Development",
        min_value=0,
        max_value=100,
        value=mixed_default,
        step=5,
        help="Live-work-play developments. Increases density and economic activity."
    )

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
    
    # Display data source info
    st.sidebar.info(f"**Data Source**: {data_source}")
    st.sidebar.info(f"**Neighborhoods**: {len(df_for_sim)}")
    
    # Run simulation
    df_sim = simulate_olympic_impacts(
        df_for_sim,
        transit_investment=transit_investment,
        community_hub_investment=community_hub_investment,
        affordable_housing=affordable_housing,
        green_spaces=green_spaces,
        mixed_use_development=mixed_use_development
    )
    stats = generate_summary_stats(df_sim)
    
    # Professional Impact Dashboard
    st.markdown("### Key Impact Indicators")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_color = "inverse" if stats['avg_rent_change'] > 0 else "normal"
        st.metric(
            "Average Rent Impact",
            f"${stats['avg_rent_change']:.0f}/mo",
            f"{stats['avg_rent_change_pct']:.1f}%",
            delta_color=delta_color,
            help="Change in average monthly rent across all neighborhoods"
        )

    with col2:
        delta_color = "inverse" if stats['avg_risk_change'] > 0 else "normal"
        st.metric(
            "Displacement Risk Change",
            f"{abs(stats['avg_risk_change']):.1f} pts",
            "increase" if stats['avg_risk_change'] > 0 else "decrease",
            delta_color=delta_color,
            help="Change in average displacement risk index (0-100 scale)"
        )

    with col3:
        total_hoods = len(df_sim)
        pct_high_risk = (stats['high_risk_neighborhoods'] / total_hoods) * 100
        st.metric(
            "High-Risk Communities",
            f"{stats['high_risk_neighborhoods']}/{total_hoods}",
            f"{pct_high_risk:.0f}% of total",
            delta_color="inverse",
            help="Neighborhoods with >70% displacement risk"
        )

    with col4:
        pct_high_rent = (stats['high_rent_increase_neighborhoods'] / total_hoods) * 100
        st.metric(
            "Significant Rent Increases",
            f"{stats['high_rent_increase_neighborhoods']}/{total_hoods}",
            f"{pct_high_rent:.0f}% of total",
            delta_color="inverse",
            help="Neighborhoods experiencing >10% rent increases"
        )

    # Add scenario comparison summary
    with st.expander("Scenario Summary", expanded=False):
        st.markdown(f"""
        **Current Scenario**: {scenario_preset}

        **Investment Profile**:
        - Transit Infrastructure: {transit_investment}%
        - Affordable Housing: {affordable_housing}%
        - Community Hubs: {community_hub_investment}%
        - Green Spaces: {green_spaces}%
        - Mixed-Use Development: {mixed_use_development}%

        **Projected Outcomes**:
        - Total neighborhoods analyzed: {len(df_sim)}
        - Average displacement risk: {df_sim['simulated_displacement_risk'].mean():.1f}%
        - Average monthly rent: ${df_sim['simulated_rent'].mean():.0f}
        - Neighborhoods protected from high risk: {total_hoods - stats['high_risk_neighborhoods']}
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Map View", "Data Table", "Charts", "Summary", "AI Assistant"])
    
    with tab1:
        st.subheader("Displacement Risk Map")
        if has_geometry:
            st.markdown("*Red areas indicate higher displacement risk. Real LA neighborhood boundaries.*")
        else:
            st.markdown("*Red areas indicate higher displacement risk. Circle size represents rent level.*")
        
        map_fig = create_choropleth_map(df_sim, has_geometry)
        st.plotly_chart(map_fig, width='stretch')
    
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
            file_name=f"planla_simulation_{transit_investment}_{community_hub_investment}.csv",
            mime="text/csv"
        )
    
    with tab3:
        st.subheader("Rent Change Analysis")
        
        chart_fig = create_rent_change_chart(df_sim)
        st.plotly_chart(chart_fig, width='stretch')
        
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
        
        # Generate LLM summary
        summary = mock_llm_summary(stats, transit_investment, community_hub_investment)
        
        st.markdown(summary)
        
        # Additional context
        st.markdown("---")
        st.markdown("### About This Simulation")
        
        st.markdown("""
        This simulation uses **real LA data** from:
        - **LA Open Data Portal**: Building permits and neighborhood data
        - **LA GeoHub**: Rent burden and demographic information
        
        The simulation models the potential impacts of Olympic investments on LA neighborhoods:
        
        - **Transit Infrastructure**: Increases rent due to improved accessibility, but may increase displacement risk
        - **Community Hubs**: Reduces displacement risk through community stability, with minimal rent impact
        
        *Note: Some fields (base_rent, median_income) are estimated for simulation compatibility when real data is unavailable.*
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
                    response = generate_ai_response(prompt, df_sim, stats, {
                        'transit': transit_investment,
                        'community_hubs': community_hub_investment,
                        'affordable_housing': affordable_housing,
                        'green_spaces': green_spaces,
                        'mixed_use': mixed_use_development
                    })
                    st.markdown(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Footer
    st.markdown("---")
    st.markdown(
        f"**PlanLA Impact Simulator** | Built with Streamlit | "
        f"Data: {data_source} | "
        f"Simulation: Olympic investment impact modeling"
    )

if __name__ == "__main__":
    main()
