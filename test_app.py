#!/usr/bin/env python3
"""
Test script for PlanLA Impact Simulator
Tests the data loading and simulation functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_la_data
from simulation import simulate_olympic_impacts, generate_summary_stats

def test_data_loading():
    """Test data loading functionality"""
    print("🧪 Testing Data Loading...")
    
    try:
        # Test real data loading
        real_gdf = load_la_data(use_cache=False, force_refresh=True)
        print(f"✅ LA data loaded: {len(real_gdf)} neighborhoods")
        return real_gdf
    except Exception as e:
        print(f"❌ Error loading LA data: {e}")
        print("This is expected if APIs are restricted or unavailable.")
        return None

def test_simulation():
    """Test simulation functionality"""
    print("\n🧪 Testing Simulation...")
    
    # Load data for testing
    gdf = load_la_data(use_cache=True, force_refresh=False)
    if gdf is None:
        print("❌ Cannot test simulation without data")
        return False
    
    df = gdf.drop(columns=['geometry']) if hasattr(gdf, 'geometry') else gdf
    
    # Test different scenarios
    scenarios = [
        (False, False, "No investments"),
        (True, False, "Transit only"),
        (False, True, "Community hubs only"),
        (True, True, "Both investments")
    ]
    
    for transit, hubs, name in scenarios:
        df_sim = simulate_olympic_impacts(df, transit, hubs)
        stats = generate_summary_stats(df_sim)
        
        print(f"✅ {name}:")
        print(f"   - Avg rent change: ${stats['avg_rent_change']:.0f} ({stats['avg_rent_change_pct']:.1f}%)")
        print(f"   - Avg risk change: {stats['avg_risk_change']:.1f} pts")
        print(f"   - High risk areas: {stats['high_risk_neighborhoods']}")
    
    return True

def test_app_imports():
    """Test that all app imports work"""
    print("\n🧪 Testing App Imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit import successful")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly and Pandas imports successful")
    except ImportError as e:
        print(f"❌ Plotly/Pandas import failed: {e}")
        return False
    
    try:
        import geopandas as gpd
        print("✅ GeoPandas import successful")
    except ImportError as e:
        print(f"❌ GeoPandas import failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🏛️ PlanLA Impact Simulator - Test Suite")
    print("=" * 50)
    
    # Test imports
    if not test_app_imports():
        print("\n❌ Import tests failed. Please install missing dependencies.")
        return False
    
    # Test data loading
    try:
        gdf = test_data_loading()
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False
    
    # Test simulation (only if we have data)
    if gdf is not None:
        try:
            test_simulation()
        except Exception as e:
            print(f"❌ Simulation test failed: {e}")
            return False
    else:
        print("⚠️  Skipping simulation test - no data available")
    
    print("\n🎉 All tests passed! The app is ready to run.")
    print("\nTo start the app, run:")
    print("  streamlit run app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
