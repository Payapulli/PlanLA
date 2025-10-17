# PlanLA Impact Simulator 🏅

An AI-powered interactive urban planning simulator that visualizes how LA 2028 Olympic infrastructure investments affect displacement risk and rent changes across 114 Los Angeles neighborhoods.

**🌐 Live Demo**: [Coming Soon - Deploy to Streamlit Cloud]

---

## 🚀 Features

### Core Capabilities
- **Neighborhood-Specific Investments**: Configure transit, affordable housing, and community hub investments for individual neighborhoods
- **Network Effects Simulation**: Investments propagate to nearby neighborhoods within 5km using spatial network analysis
- **Budget Constraints**: $500M total budget with real-time validation and warnings
- **AI Planning Assistant**: Chat with an intelligent AI (Hugging Face Llama 3.1-8B) that analyzes your current simulation and provides context-aware recommendations
- **Real LA Data**: 114 LA Times neighborhoods with US Census rent burden data from 2,498 census tracts

### Interactive Visualizations
- **Map View**: Choropleth maps with real LA neighborhood boundaries showing displacement risk
- **Data Table**: Sortable neighborhood metrics with customizable columns
- **Charts**: Rent change analysis with color-coded bars (green = decrease, red = increase)
- **Summary**: Automated impact analysis with key findings and recommendations
- **AI Assistant**: Conversational interface for strategy optimization

---

## 📊 Data Sources

### Real LA Data
- **LA Times Neighborhood Boundaries**: 114 official LA neighborhoods with polygon geometries
- **US Census Bureau**: American Community Survey (ACS) 5-Year 2022 rent burden data
- **Spatial Aggregation**: Census tract data spatially joined to neighborhoods using geometric intersections
- **Olympic Venues**: Distance calculated to LA Memorial Coliseum, Crypto.com Arena, and SoFi Stadium

### Synthetic Data (Fallback)
- **Building Permits**: Estimated permit density when API unavailable
- **Base Rent**: Estimated from rent burden + median income when direct data unavailable

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, GeoPandas, NumPy
- **Visualization**: Plotly
- **Geospatial**: Shapely, Fiona, GDAL
- **Network Analysis**: NetworkX, SciPy
- **AI Assistant**: Hugging Face Transformers (Llama 3.1-8B), OpenAI GPT-4 (fallback)
- **APIs**: Requests (LA data), python-dotenv (config)

---

## 🚀 Quick Start

### Option 1: Use the startup script (easiest)
```bash
./run.sh
```

### Option 2: Manual setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will be available at `http://localhost:8501`

---

## 🎛️ How to Use

### 1. Select a Neighborhood
- Use the dropdown in the sidebar to select a neighborhood
- View current rent, displacement risk, and investment status

### 2. Configure Investments
- **Transit Infrastructure** ($10M increments): Metro lines, bus rapid transit
  - Effect: +5-15% rent, +5 points displacement risk
- **Affordable Housing** ($10M increments): Deed-restricted affordable units
  - Effect: -5% rent, -40% displacement risk reduction
- **Community Hubs** ($5M increments): Community centers, social services
  - Effect: +3% rent, -25% displacement risk reduction

### 3. Apply Investments
- Click "Apply Investment" to save changes
- Budget is validated in real-time
- Investments propagate to neighbors within 5km (30% spillover strength)

### 4. Explore Results
- 🗺️ **Map View**: Interactive choropleth showing displacement risk
- 📊 **Data Table**: Detailed metrics for all 114 neighborhoods
- 📈 **Charts**: Rent change analysis by neighborhood
- 📝 **Summary**: Automated impact analysis
- 🤖 **AI Assistant**: Ask questions like:
  - "How can I reduce displacement in Downtown LA?"
  - "What's the optimal investment strategy?"
  - "Should I invest more in transit or affordable housing?"

### 5. Budget Management
- **Total Budget**: $500M (configurable)
- **Budget Tracking**: Live display in sidebar
- **Warnings**: Alerts when budget is low (<$50M) or exceeded

---

## 📁 Project Structure

```
PlanLA/
├── app.py                     # Main Streamlit application
├── data_loader.py             # Real LA data ingestion from Census + LA Times
├── simulation.py              # Olympic investment impact simulation
├── network_effects.py         # Spatial network analysis & spillover calculations
├── config.py                  # API keys configuration (optional)
├── requirements.txt           # Python dependencies
├── packages.txt               # System dependencies (for Streamlit Cloud)
├── .streamlit/config.toml     # Streamlit theme configuration
├── olympics-image.png         # LA 2028 Olympics logo
├── test_app.py                # Test suite
├── run.sh                     # Startup script
├── README.md                  # This file
├── .gitignore                 # Git ignore rules
├── CLAUDE.md                  # AI assistant guidance
└── la_impact_data.geojson     # Cached real data (auto-generated)
```

---

## 🔧 Simulation Logic

### Displacement Risk Calculation

Displacement risk is a composite score (0-100) calculated from:

1. **Rent Burden (30% weight)**:
   ```
   rent_burden = (monthly_rent × 12) / median_household_income
   ```

2. **Development Pressure (20% weight)**:
   ```
   development_pressure = building_permit_density / 10
   ```

3. **Olympic Proximity (50% weight)**:
   ```
   proximity_factor = max(0, (50km - distance_to_venues) / 50km)
   ```

**Combined Formula**:
```python
displacement_risk = (
    rent_burden × 30 +
    development_pressure × 20 +
    proximity_factor × 50
)
# Clipped to 0-100 range
```

### Network Effects Algorithm

Investments propagate to nearby neighborhoods:

```python
# 1. Build weighted graph based on geographic proximity
for each pair of neighborhoods:
    distance = haversine(neighborhood_A, neighborhood_B)
    if distance <= 5km:
        weight = 1 / (1 + distance)  # Inverse distance weighting
        add_edge(A, B, weight)

# 2. Calculate spillover effects
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

---

## 🧪 Testing

Run the comprehensive test suite:
```bash
python3 test_app.py
```

Tests include:
- ✅ Import verification
- ✅ Data loading (real + mock)
- ✅ Simulation scenarios
- ✅ Network effects
- ✅ Error handling

---

## 🚀 Deployment

### Deploy to Streamlit Community Cloud (Free)

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select repository: `Payapulli/PlanLA`
   - Main file path: `app.py`
   - Click "Deploy"

3. **Optional: Configure Secrets** (for AI features):
   - In Streamlit Cloud dashboard, go to App Settings → Secrets
   - Add API keys:
     ```toml
     HF_API_KEY = "your_huggingface_api_key_here"
     OPENAI_API_KEY = "your_openai_api_key_here"  # Optional fallback
     ```

4. **Your app will be live at**: `https://planla.streamlit.app` (or similar)

### System Requirements
- The `packages.txt` file installs required system dependencies (GDAL, spatial libraries)
- The `.streamlit/config.toml` applies your custom theme (orange highlights, light gray buttons)
- All Python dependencies are in `requirements.txt`

---

## 🤖 AI Assistant Technical Details

### Three-Tier Intelligence System

1. **Primary: Hugging Face Llama 3.1-8B**
   - Free, open-source language model
   - Configured for urban planning expertise
   - Analyzes current simulation state in real-time

2. **Fallback: OpenAI GPT-4**
   - Premium model for complex queries
   - Requires API key (optional)
   - Used when Hugging Face unavailable

3. **Always Available: Rule-Based Intelligence**
   - Pattern matching for common scenarios
   - Statistical analysis of simulation data
   - Pre-programmed urban planning heuristics

### Context Awareness
The AI receives:
- Current investments per neighborhood
- Simulation results (rent changes, displacement risk)
- Top 3 most at-risk neighborhoods
- Network spillover effects
- Budget status

---

## 🔮 Future Enhancements

### Data Integration
- [ ] Additional LA data sources (Zillow, Census TIGER)
- [ ] Real-time data updates
- [ ] Historical trend analysis (2010-2024)

### Simulation Improvements
- [ ] Machine learning models for rent prediction
- [ ] Temporal modeling (2024-2030 timeline)
- [ ] Multi-city support (other Olympic host cities)
- [ ] Climate impact factors

### Technical Features
- [ ] User authentication (save scenarios)
- [ ] Scenario comparison (A/B testing)
- [ ] Export to PDF reports
- [ ] Mobile-responsive design

---

## 📝 Notes

- **API Restrictions**: US Census API may have rate limits; caching mitigates this
- **Mock Data**: Always available for demonstration when APIs unavailable
- **Geographic Data**: Uses real LA Times neighborhood boundaries with 114 polygons
- **AI Features**: Work without API keys via rule-based fallback

---

## 🤝 Contributing

This is a prototype for educational/research purposes. Contributions welcome for:
- Additional data sources
- Improved simulation models
- UI/UX enhancements
- Documentation improvements

---

## 📄 License

Educational/Research Use - See project requirements for specific usage terms.

---

## 📧 Contact

For questions or feedback about this project, please open an issue on GitHub.

**Built for the LA 2028 Olympics 🏅**
