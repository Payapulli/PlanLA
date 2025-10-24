# PlanLA Impact Simulator

An AI-powered interactive urban planning simulator that visualizes how LA 2028 Olympic infrastructure investments affect displacement risk and rent changes across 114 Los Angeles neighborhoods.

---

## Features

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

## Data Sources

### Real LA Data
- **LA Times Neighborhood Boundaries**: 114 official LA neighborhoods with polygon geometries
- **US Census Bureau**: American Community Survey (ACS) 5-Year 2022 rent burden data
- **Spatial Aggregation**: Census tract data spatially joined to neighborhoods using geometric intersections
- **Olympic Venues**: Distance calculated to LA Memorial Coliseum, Crypto.com Arena, and SoFi Stadium

### Synthetic Data (Fallback)
- **Building Permits**: Estimated permit density when API unavailable
- **Base Rent**: Estimated from rent burden + median income when direct data unavailable

---

## Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, GeoPandas, NumPy
- **Visualization**: Plotly
- **Geospatial**: Shapely, Fiona, GDAL
- **Network Analysis**: NetworkX, SciPy
- **AI Assistant**: Hugging Face Transformers (Llama 3.1-8B), OpenAI GPT-4 (fallback)
- **APIs**: Requests (LA data), python-dotenv (config)

---

## Quick Start

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

## How to Use

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
- **Map View**: Interactive choropleth showing displacement risk
- **Data Table**: Detailed metrics for all 114 neighborhoods
- **Charts**: Rent change analysis by neighborhood
- **Summary**: Automated impact analysis
- **AI Assistant**: Ask questions like:
  - "How can I reduce displacement in Downtown LA?"
  - "What's the optimal investment strategy?"
  - "Should I invest more in transit or affordable housing?"

### 5. Budget Management
- **Total Budget**: $500M (configurable)
- **Budget Tracking**: Live display in sidebar
- **Warnings**: Alerts when budget is low (<$50M) or exceeded

---