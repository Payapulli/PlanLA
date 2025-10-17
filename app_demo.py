"""
PlanLA Impact Simulator - Demo Version with Neighborhood-Specific Investments
Click on neighborhoods to invest, watch effects ripple through the network!
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_loader import load_la_data
from simulation import simulate_localized_investments, generate_summary_stats
from network_effects import build_neighborhood_network

# Page configuration
st.set_page_config(
    page_title="PlanLA Impact Simulator - Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF;
    }

    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #5A6C7D;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        font-size: 1.1rem;
        color: #5A6C7D;
        text-align: center;
        margin-bottom: 2rem;
        opacity: 0.7;
    }

    .stButton > button {
        background-color: #5A6C7D;
        color: #FFFFFF;
        border: none;
        border-radius: 0.375rem;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #4a5b6b;
    }
</style>
""", unsafe_allow_html=True)

def create_interactive_map(gdf, selected_neighborhood=None):
    """Create clickable choropleth map showing displacement risk."""

    # Convert GeoDataFrame to GeoJSON
    gdf_json = gdf.to_json()

    # Highlight selected neighborhood
    if selected_neighborhood:
        colors = ['#FFD700' if hood == selected_neighborhood else gdf.loc[gdf['neighborhood'] == hood, 'simulated_displacement_risk'].values[0]
                  for hood in gdf['neighborhood']]
        color_scale = None
    else:
        colors = gdf['simulated_displacement_risk']
        color_scale = 'Reds'

    fig = px.choropleth_mapbox(
        gdf,
        geojson=gdf_json,
        locations=gdf.index,
        color=colors if selected_neighborhood else 'simulated_displacement_risk',
        hover_data={
            'neighborhood': True,
            'simulated_displacement_risk': ':.1f',
            'spillover_rent_impact': ':.2f',
            'spillover_risk_impact': ':.2f',
            'rent_change_pct': ':.1f'
        },
        color_continuous_scale=color_scale,
        mapbox_style="open-street-map",
        center={"lat": 34.0522, "lon": -118.2437},
        zoom=9,
        title="Click on a neighborhood to invest"
    )

    fig.update_layout(
        height=600,
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        clickmode='event+select'
    )

    return fig

def main():
    """Main app function."""

    # Header
    st.markdown('<h1 class="main-header">PlanLA Impact Simulator - Interactive Demo</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Click neighborhoods to invest → Watch network effects ripple out</p>', unsafe_allow_html=True)

    # Initialize session state
    if 'investments' not in st.session_state:
        st.session_state.investments = {}
    if 'selected_neighborhood' not in st.session_state:
        st.session_state.selected_neighborhood = None
    if 'network' not in st.session_state:
        st.session_state.network = None

    # Load data
    with st.spinner("Loading LA data..."):
        df = load_la_data(use_cache=True, force_refresh=False)

        if hasattr(df, 'geometry'):
            df_for_sim = pd.DataFrame(df.drop(columns=['geometry']))
        else:
            df_for_sim = df

    # Build network once
    if st.session_state.network is None:
        with st.spinner("Building neighborhood network..."):
            st.session_state.network = build_neighborhood_network(df_for_sim, distance_threshold=5.0)
            st.success(f"Built network with {len(st.session_state.network)} neighborhoods!")

    # Run simulation with current investments
    df_sim = simulate_localized_investments(
        df_for_sim,
        st.session_state.investments,
        st.session_state.network
    )

    # Merge geometry back in
    if hasattr(df, 'geometry'):
        df_sim['geometry'] = df['geometry'].values
        import geopandas as gpd
        gdf_sim = gpd.GeoDataFrame(df_sim, geometry='geometry')
    else:
        gdf_sim = df_sim

    stats = generate_summary_stats(df_sim)

    # Layout: Map on left, Controls on right
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Neighborhood Map")

        # Display the map
        map_fig = create_interactive_map(gdf_sim, st.session_state.selected_neighborhood)
        st.plotly_chart(map_fig, use_container_width=True)

    with col2:
        st.subheader("Investment Panel")

        # Neighborhood selector (dropdown instead of click - more reliable)
        st.markdown("**Select a neighborhood:**")
        selected = st.selectbox(
            "Choose neighborhood",
            options=['None'] + sorted(df_sim['neighborhood'].tolist()),
            index=0 if st.session_state.selected_neighborhood is None else
                  sorted(df_sim['neighborhood'].tolist()).index(st.session_state.selected_neighborhood) + 1,
            label_visibility="collapsed"
        )

        if selected != 'None':
            st.session_state.selected_neighborhood = selected
        else:
            st.session_state.selected_neighborhood = None

        if st.session_state.selected_neighborhood:
            hood = st.session_state.selected_neighborhood
            st.success(f"Selected: **{hood}**")

            # Show current stats for this neighborhood
            hood_data = df_sim[df_sim['neighborhood'] == hood].iloc[0]

            st.metric("Current Rent", f"${hood_data['base_rent']:.0f}/mo")
            st.metric("Displacement Risk", f"{hood_data['simulated_displacement_risk']:.1f}%")

            st.markdown("---")
            st.markdown("### Make an Investment")

            # Investment buttons
            col_a, col_b = st.columns(2)

            with col_a:
                if st.button("🚇 Transit\n$50M", key="transit"):
                    if hood not in st.session_state.investments:
                        st.session_state.investments[hood] = {'transit': 0, 'affordable': 0, 'community': 0}
                    st.session_state.investments[hood]['transit'] += 50_000_000
                    st.rerun()

                if st.button("🏘️ Affordable\n$30M", key="affordable"):
                    if hood not in st.session_state.investments:
                        st.session_state.investments[hood] = {'transit': 0, 'affordable': 0, 'community': 0}
                    st.session_state.investments[hood]['affordable'] += 30_000_000
                    st.rerun()

            with col_b:
                if st.button("🏢 Community\n$10M", key="community"):
                    if hood not in st.session_state.investments:
                        st.session_state.investments[hood] = {'transit': 0, 'affordable': 0, 'community': 0}
                    st.session_state.investments[hood]['community'] += 10_000_000
                    st.rerun()

                if st.button("🗑️ Clear All", key="clear"):
                    st.session_state.investments = {}
                    st.session_state.selected_neighborhood = None
                    st.rerun()

            # Show current investments in this neighborhood
            if hood in st.session_state.investments:
                st.markdown("---")
                st.markdown("### Current Investments Here")
                inv = st.session_state.investments[hood]
                if inv['transit'] > 0:
                    st.write(f"🚇 Transit: ${inv['transit']/1_000_000:.0f}M")
                if inv['affordable'] > 0:
                    st.write(f"🏘️ Affordable: ${inv['affordable']/1_000_000:.0f}M")
                if inv['community'] > 0:
                    st.write(f"🏢 Community: ${inv['community']/1_000_000:.0f}M")

            # Show spillover effects
            if hood_data['spillover_rent_impact'] != 0 or hood_data['spillover_risk_impact'] != 0:
                st.markdown("---")
                st.markdown("### Network Spillover Effects")
                st.write(f"💸 Rent spillover: ${hood_data['spillover_rent_impact']:.2f}")
                st.write(f"📊 Risk spillover: {hood_data['spillover_risk_impact']:.2f} pts")

        else:
            st.info("☝️ Select a neighborhood from the dropdown above to start investing!")
            st.markdown("### How It Works")
            st.markdown("""
            1. **Select** a neighborhood from dropdown
            2. **Invest** using the buttons below
            3. **Watch** effects ripple to nearby neighborhoods

            **Investment Effects:**
            - 🚇 Transit: +rent, +risk (gentrification)
            - 🏘️ Affordable: -rent, -risk (protection)
            - 🏢 Community: -risk (stability)

            **Network Effects:**
            - Investments spread to neighbors within 5km
            - Closer neighborhoods = stronger spillover
            - 30% spillover strength
            - Gold = selected neighborhood
            """)

    # Summary metrics at bottom
    st.markdown("---")
    st.subheader("Overall Impact")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_invested = sum(
            inv['transit'] + inv['affordable'] + inv['community']
            for inv in st.session_state.investments.values()
        )
        st.metric("Total Invested", f"${total_invested/1_000_000:.0f}M")

    with col2:
        st.metric("Avg Rent Change", f"${stats['avg_rent_change']:.0f}", f"{stats['avg_rent_change_pct']:.1f}%")

    with col3:
        st.metric("Avg Risk Change", f"{stats['avg_risk_change']:.1f} pts")

    with col4:
        st.metric("High Risk Areas", f"{stats['high_risk_neighborhoods']}/{len(df_sim)}")

if __name__ == "__main__":
    main()
