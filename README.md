# PlanLA Impact Simulator 🏛️

An AI-powered interactive urban planning simulator that visualizes how Olympic-related investments affect displacement risk and rent changes across Los Angeles neighborhoods.

## 🚀 Features

- **5 Investment Types** with intensity sliders (0-100%): Transit Infrastructure, Community Hubs, Affordable Housing, Green Spaces, Mixed-Use Development
- **AI Planning Assistant**: Chat with an intelligent AI that analyzes your simulation and provides recommendations
- **Real LA Data**: 15 real LA neighborhoods with actual coordinates and demographics
- **Interactive Maps**: Choropleth maps showing displacement risk by neighborhood
- **Real-Time Simulation**: Instant visualization of investment impacts
- **Data Export**: Download simulation results as CSV

## 📊 Data Sources

### Real LA Data Only
- **LA Open Data Portal**: Building permits and neighborhood data
- **LA GeoHub**: Rent burden and demographic information
- **Intelligent Caching**: Automatically saves data locally for offline use
- **No Fallback**: App requires real data to function - fails gracefully if APIs are unavailable

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, GeoPandas
- **Visualization**: Plotly
- **APIs**: Requests (for LA data sources)
- **Geospatial**: Shapely, Fiona

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

### Option 3: Test first
```bash
# Run test suite
python3 test_app.py

# Then start the app
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 🎛️ How to Use

1. **Set Investments**: Enable/disable transit infrastructure and community hubs in the sidebar
2. **Explore Results**: 
   - 🗺️ **Map View**: Interactive choropleth map with real LA neighborhood boundaries
   - 📊 **Data Table**: Sortable neighborhood metrics including permit density and rent burden
   - 📈 **Charts**: Rent change analysis by neighborhood
   - 📝 **Summary**: AI-generated insights and recommendations

## 📁 Project Structure

```
PlanLA/
├── app.py                 # Main Streamlit application
├── data_loader.py         # Real LA data ingestion + mock data
├── simulation.py          # Olympic investment impact simulation
├── requirements.txt       # Python dependencies
├── test_app.py           # Test suite
├── run.sh                # Startup script
├── README.md             # This file
└── la_impact_data.geojson # Cached real data (auto-generated)
```

## 🔧 Simulation Logic

The simulator models two types of Olympic investments:

### Transit Infrastructure
- **Effect**: Increases rent (5-15% based on proximity to Olympic sites)
- **Risk**: Slightly increases displacement risk due to gentrification
- **Modeling**: Distance-based rent multiplier + 5-point risk increase

### Community Hubs
- **Effect**: Reduces displacement risk (20% reduction)
- **Rent Impact**: Minimal rent increase (2%) due to improved amenities
- **Modeling**: Risk reduction proportional to baseline risk

## 🌐 Data Integration Details

### Real Data Sources
- **Building Permits**: `https://data.lacity.org/resource/nbyu-2ha9.json`
- **Rent Burden**: `https://geohub.lacity.org/datasets/rent-and-mortgage-burdened-households-in-los-angeles.geojson`

### API Handling
- Graceful fallback to mock data when APIs are restricted
- Local caching for offline use
- Error handling for rate limits and access restrictions

### Data Processing
- Aggregates building permits by neighborhood
- Merges with rent burden data
- Calculates Olympic venue distances
- Adds simulation-compatible fields

## 🧪 Testing

Run the comprehensive test suite:
```bash
python3 test_app.py
```

Tests include:
- ✅ Import verification
- ✅ Data loading (real + mock)
- ✅ Simulation scenarios
- ✅ Error handling

## 🚀 Deployment

### Local Development
- Use `run.sh` for easy startup
- Virtual environment included
- All dependencies managed

### Future Deployment Options
- **Vercel**: Streamlit apps supported
- **Docker**: Containerization ready
- **Cloud**: AWS/GCP/Azure compatible

## 🔮 Future Enhancements

### Data Integration
- [ ] Additional LA data sources (Census, Zillow, etc.)
- [ ] Real-time data updates
- [ ] Historical trend analysis

### Simulation Improvements
- [ ] Machine learning models for rent prediction
- [ ] Temporal modeling (2024-2030 timeline)
- [ ] Cascading neighborhood effects
- [ ] Multiple investment types

### Technical Features
- [ ] Real LLM integration (OpenAI API)
- [ ] FastAPI backend
- [ ] Database integration
- [ ] User authentication

## 📝 Notes

- **API Restrictions**: Many public APIs have rate limits or require authentication
- **Mock Data**: Always available for demonstration and development
- **Caching**: Real data is cached locally for performance
- **Geographic Data**: Uses LA neighborhood boundaries when available

## 🤝 Contributing

This is a prototype for educational/research purposes. Contributions welcome for:
- Additional data sources
- Improved simulation models
- UI/UX enhancements
- Documentation improvements

## 📄 License

Educational/Research Use - See project requirements for specific usage terms.
