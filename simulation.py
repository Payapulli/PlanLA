"""
Simulation module for PlanLA Impact Simulator.
Contains logic for calculating Olympic investment impacts on neighborhoods.
Updated to support neighborhood-specific investments with network effects.
"""

import pandas as pd
import numpy as np
from network_effects import build_neighborhood_network, calculate_spillover_effects

def calculate_displacement_risk(base_rent, median_income, permit_density, distance_to_olympic_site):
    """
    Calculate baseline displacement risk for a neighborhood.
    
    Args:
        base_rent (float): Current average rent
        median_income (float): Median household income
        permit_density (float): Building permit density
        distance_to_olympic_site (float): Distance to Olympic venues (km)
    
    Returns:
        float: Displacement risk score (0-100)
    """
    # Rent burden: higher rent relative to income increases risk
    rent_burden = (base_rent * 12) / median_income
    
    # Development pressure: more permits increase risk
    development_pressure = permit_density / 10  # Normalize
    
    # Proximity factor: closer to Olympic sites = higher risk
    proximity_factor = max(0, (50 - distance_to_olympic_site) / 50)
    
    # Calculate risk score
    risk_score = (rent_burden * 30 + development_pressure * 20 + proximity_factor * 50)
    
    return min(100, max(0, risk_score))

def simulate_olympic_impacts(df, transit_investment=0, community_hub_investment=0,
                            affordable_housing=0, green_spaces=0, mixed_use_development=0):
    """
    Simulate the impact of Olympic investments on neighborhoods.

    Args:
        df (pd.DataFrame): Neighborhood data
        transit_investment (float): Transit investment intensity (0-100)
        community_hub_investment (float): Community hub investment intensity (0-100)
        affordable_housing (float): Affordable housing investment intensity (0-100)
        green_spaces (float): Green spaces investment intensity (0-100)
        mixed_use_development (float): Mixed-use development intensity (0-100)

    Returns:
        pd.DataFrame: Updated neighborhood data with simulated impacts
    """
    df_sim = df.copy()

    # Calculate baseline displacement risk
    df_sim['baseline_displacement_risk'] = df_sim.apply(
        lambda row: calculate_displacement_risk(
            row['base_rent'],
            row['median_income'],
            row['permit_density'],
            row['distance_to_olympic_site']
        ), axis=1
    )

    # Initialize simulated values
    df_sim['simulated_rent'] = df_sim['base_rent']
    df_sim['simulated_displacement_risk'] = df_sim['baseline_displacement_risk']

    # Transit investment impact (scales with intensity)
    if transit_investment > 0:
        intensity = transit_investment / 100.0
        # Transit investment increases rent by 5-15% based on distance to Olympic sites
        # Closer neighborhoods see higher rent increases
        transit_multiplier = 1 + intensity * (0.15 - (df_sim['distance_to_olympic_site'] / 50) * 0.1)
        df_sim['simulated_rent'] *= transit_multiplier

        # Transit also increases displacement risk due to gentrification
        df_sim['simulated_displacement_risk'] += 5 * intensity

    # Community hub investment impact (scales with intensity)
    if community_hub_investment > 0:
        intensity = community_hub_investment / 100.0
        # Community hubs reduce displacement risk by providing stability
        # Effect is stronger in neighborhoods with higher baseline risk
        risk_reduction = df_sim['baseline_displacement_risk'] * 0.25 * intensity  # Up to 25% reduction
        df_sim['simulated_displacement_risk'] -= risk_reduction

        # Community hubs may slightly increase rent due to improved amenities
        df_sim['simulated_rent'] *= (1 + 0.03 * intensity)

    # Affordable housing impact (scales with intensity)
    if affordable_housing > 0:
        intensity = affordable_housing / 100.0
        # Affordable housing significantly reduces displacement risk
        risk_reduction = df_sim['baseline_displacement_risk'] * 0.4 * intensity  # Up to 40% reduction
        df_sim['simulated_displacement_risk'] -= risk_reduction

        # Slight decrease in market-rate rents due to increased supply
        df_sim['simulated_rent'] *= (1 - 0.05 * intensity)

    # Green spaces impact (scales with intensity)
    if green_spaces > 0:
        intensity = green_spaces / 100.0
        # Green spaces increase rent due to desirability
        df_sim['simulated_rent'] *= (1 + 0.08 * intensity)

        # Moderate impact on displacement risk (can go either way)
        df_sim['simulated_displacement_risk'] += 3 * intensity

    # Mixed-use development impact (scales with intensity)
    if mixed_use_development > 0:
        intensity = mixed_use_development / 100.0
        # Mixed-use increases rent significantly
        df_sim['simulated_rent'] *= (1 + 0.12 * intensity)

        # Can reduce displacement if done with community benefits
        risk_reduction = df_sim['baseline_displacement_risk'] * 0.15 * intensity
        df_sim['simulated_displacement_risk'] -= risk_reduction
    
    # Ensure values stay within reasonable bounds
    df_sim['simulated_rent'] = df_sim['simulated_rent'].clip(lower=1000, upper=6000)
    df_sim['simulated_displacement_risk'] = df_sim['simulated_displacement_risk'].clip(lower=0, upper=100)
    
    # Calculate changes
    df_sim['rent_change'] = df_sim['simulated_rent'] - df_sim['base_rent']
    df_sim['rent_change_pct'] = (df_sim['rent_change'] / df_sim['base_rent']) * 100
    df_sim['risk_change'] = df_sim['simulated_displacement_risk'] - df_sim['baseline_displacement_risk']
    
    return df_sim

def generate_summary_stats(df_sim):
    """
    Generate summary statistics for the simulation results.
    
    Args:
        df_sim (pd.DataFrame): Simulated neighborhood data
    
    Returns:
        dict: Summary statistics
    """
    stats = {
        'total_neighborhoods': len(df_sim),
        'avg_rent_change': df_sim['rent_change'].mean(),
        'avg_rent_change_pct': df_sim['rent_change_pct'].mean(),
        'avg_risk_change': df_sim['risk_change'].mean(),
        'high_risk_neighborhoods': len(df_sim[df_sim['simulated_displacement_risk'] > 70]),
        'high_rent_increase_neighborhoods': len(df_sim[df_sim['rent_change_pct'] > 10])
    }
    
    return stats

def simulate_localized_investments(df, investments_by_neighborhood, network=None):
    """
    Simulate impacts of neighborhood-specific investments with network spillover effects.

    Args:
        df (pd.DataFrame): Neighborhood data
        investments_by_neighborhood (dict): {neighborhood: {'transit': $X, 'affordable': $Y, 'community': $Z}}
        network (dict): Neighborhood adjacency network (optional, will build if not provided)

    Returns:
        pd.DataFrame: Updated neighborhood data with direct and spillover impacts
    """
    df_sim = df.copy()

    # Calculate baseline displacement risk
    df_sim['baseline_displacement_risk'] = df_sim.apply(
        lambda row: calculate_displacement_risk(
            row['base_rent'],
            row['median_income'],
            row['permit_density'],
            row['distance_to_olympic_site']
        ), axis=1
    )

    # Initialize simulated values (no changes yet)
    df_sim['simulated_rent'] = df_sim['base_rent']
    df_sim['simulated_displacement_risk'] = df_sim['baseline_displacement_risk']

    # Build neighborhood network if not provided
    if network is None:
        network = build_neighborhood_network(df, distance_threshold=5.0)

    # Apply network effects to calculate spillover
    df_with_spillover = calculate_spillover_effects(
        df_sim,
        investments_by_neighborhood,
        network,
        spillover_strength=0.3
    )

    # Apply spillover impacts to rent and risk
    df_with_spillover['simulated_rent'] += df_with_spillover['spillover_rent_impact']
    df_with_spillover['simulated_displacement_risk'] += df_with_spillover['spillover_risk_impact']

    # Ensure values stay within reasonable bounds
    df_with_spillover['simulated_rent'] = df_with_spillover['simulated_rent'].clip(lower=1000, upper=6000)
    df_with_spillover['simulated_displacement_risk'] = df_with_spillover['simulated_displacement_risk'].clip(lower=0, upper=100)

    # Calculate changes
    df_with_spillover['rent_change'] = df_with_spillover['simulated_rent'] - df_with_spillover['base_rent']
    df_with_spillover['rent_change_pct'] = (df_with_spillover['rent_change'] / df_with_spillover['base_rent']) * 100
    df_with_spillover['risk_change'] = df_with_spillover['simulated_displacement_risk'] - df_with_spillover['baseline_displacement_risk']

    return df_with_spillover

def mock_llm_summary(stats, transit_investment, community_hub_investment):
    """
    Generate a mock LLM summary of the simulation results.
    In a real implementation, this would call an LLM API.
    
    Args:
        stats (dict): Summary statistics
        transit_investment (bool): Whether transit investment is active
        community_hub_investment (bool): Whether community hub investment is active
    
    Returns:
        str: Generated summary text
    """
    summary_parts = []
    
    # Opening
    summary_parts.append("## Impact Analysis Summary")
    summary_parts.append("")
    
    # Investment status
    investments = []
    if transit_investment:
        investments.append("transit infrastructure")
    if community_hub_investment:
        investments.append("community hubs")
    
    if investments:
        summary_parts.append(f"With investments in {', '.join(investments)}, the simulation shows:")
    else:
        summary_parts.append("Without Olympic investments, the baseline scenario shows:")
    
    summary_parts.append("")
    
    # Key findings
    if stats['avg_rent_change'] > 0:
        summary_parts.append(f"• **Rent increases**: Average rent rises by ${stats['avg_rent_change']:.0f} ({stats['avg_rent_change_pct']:.1f}%)")
    else:
        summary_parts.append(f"• **Rent stability**: Average rent changes by ${stats['avg_rent_change']:.0f} ({stats['avg_rent_change_pct']:.1f}%)")
    
    if stats['avg_risk_change'] > 0:
        summary_parts.append(f"• **Displacement risk**: Increases by {stats['avg_risk_change']:.1f} points on average")
    else:
        summary_parts.append(f"• **Displacement risk**: Decreases by {abs(stats['avg_risk_change']):.1f} points on average")
    
    summary_parts.append(f"• **High-risk areas**: {stats['high_risk_neighborhoods']} neighborhoods face displacement risk >70%")
    summary_parts.append(f"• **Significant rent increases**: {stats['high_rent_increase_neighborhoods']} neighborhoods see >10% rent increases")
    
    summary_parts.append("")
    
    # Recommendations
    summary_parts.append("### Recommendations")
    if stats['high_risk_neighborhoods'] > 5:
        summary_parts.append("• Consider additional community hub investments in high-risk areas")
    if stats['high_rent_increase_neighborhoods'] > 3:
        summary_parts.append("• Implement rent stabilization measures in affected neighborhoods")
    
    summary_parts.append("• Monitor displacement patterns and adjust policies accordingly")
    
    return "\n".join(summary_parts)

if __name__ == "__main__":
    # Test the simulation
    from data_loader import load_neighborhood_data
    
    df = load_neighborhood_data()
    df_sim = simulate_olympic_impacts(df, transit_investment=True, community_hub_investment=True)
    
    print("Simulation results:")
    print(df_sim[['neighborhood', 'base_rent', 'simulated_rent', 'rent_change_pct', 
                  'baseline_displacement_risk', 'simulated_displacement_risk', 'risk_change']].head())
    
    stats = generate_summary_stats(df_sim)
    print(f"\nSummary stats: {stats}")
    
    summary = mock_llm_summary(stats, True, True)
    print(f"\nGenerated summary:\n{summary}")



