"""
Network effects module for PlanLA Impact Simulator.
Models how investments in one neighborhood affect surrounding neighborhoods using graph algorithms.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

def build_neighborhood_network(df, distance_threshold=5.0):
    """
    Build a weighted graph of neighborhoods based on proximity.

    Args:
        df (pd.DataFrame): Neighborhood data with lat/lon coordinates
        distance_threshold (float): Max distance (km) to create an edge

    Returns:
        dict: Adjacency dictionary {neighborhood: [(neighbor, weight), ...]}
    """
    network = {}

    # Get coordinates
    coords = df[['lat', 'lon']].values
    neighborhoods = df['neighborhood'].values

    # Calculate pairwise distances (using haversine approximation)
    distances = cdist(coords, coords, metric='euclidean') * 111  # rough km conversion

    for i, hood in enumerate(neighborhoods):
        neighbors = []
        for j, other_hood in enumerate(neighborhoods):
            if i != j and distances[i, j] < distance_threshold:
                # Weight inversely proportional to distance (closer = stronger effect)
                weight = 1.0 / (distances[i, j] + 0.1)  # +0.1 to avoid division by zero
                neighbors.append((other_hood, weight))

        # Normalize weights so they sum to 1
        if neighbors:
            total_weight = sum(w for _, w in neighbors)
            neighbors = [(n, w/total_weight) for n, w in neighbors]

        network[hood] = neighbors

    return network

def calculate_spillover_effects(df, investments, network, spillover_strength=0.3):
    """
    Calculate how investments in one neighborhood spill over to neighbors.
    Uses graph diffusion algorithm.

    Args:
        df (pd.DataFrame): Neighborhood data
        investments (dict): {neighborhood: {'transit': X, 'affordable': Y, ...}}
        network (dict): Neighborhood adjacency network
        spillover_strength (float): How much of the effect propagates (0-1)

    Returns:
        pd.DataFrame: Updated neighborhood data with spillover effects
    """
    df_with_spillover = df.copy()

    # Initialize spillover values
    df_with_spillover['spillover_rent_impact'] = 0.0
    df_with_spillover['spillover_risk_impact'] = 0.0

    # For each neighborhood with investment
    for hood, investment in investments.items():
        if hood not in network:
            continue

        # Calculate direct impact on this neighborhood
        direct_rent_impact = 0
        direct_risk_impact = 0

        if investment.get('transit', 0) > 0:
            direct_rent_impact += investment['transit'] * 0.001  # $1 per unit investment
            direct_risk_impact += investment['transit'] * 0.0001  # 0.01 risk per unit

        if investment.get('affordable', 0) > 0:
            direct_rent_impact -= investment['affordable'] * 0.0005
            direct_risk_impact -= investment['affordable'] * 0.0002

        if investment.get('community', 0) > 0:
            direct_risk_impact -= investment['community'] * 0.00015

        # Propagate to neighbors
        for neighbor_hood, edge_weight in network[hood]:
            propagated_rent = direct_rent_impact * edge_weight * spillover_strength
            propagated_risk = direct_risk_impact * edge_weight * spillover_strength

            # Add to neighbor's spillover
            neighbor_idx = df_with_spillover[df_with_spillover['neighborhood'] == neighbor_hood].index
            if len(neighbor_idx) > 0:
                df_with_spillover.loc[neighbor_idx[0], 'spillover_rent_impact'] += propagated_rent
                df_with_spillover.loc[neighbor_idx[0], 'spillover_risk_impact'] += propagated_risk

    return df_with_spillover

def find_optimal_investment_allocation(df, network, total_budget=500_000_000,
                                       target='minimize_displacement'):
    """
    Find optimal investment allocation across neighborhoods using greedy algorithm.

    Args:
        df (pd.DataFrame): Neighborhood data
        network (dict): Neighborhood adjacency network
        total_budget (float): Total budget in dollars
        target (str): 'minimize_displacement' or 'maximize_equity'

    Returns:
        dict: Optimal investment allocation {neighborhood: {'transit': X, 'affordable': Y}}
    """
    # Greedy algorithm: Iteratively invest in neighborhood with highest ROI

    remaining_budget = total_budget
    investments = {hood: {'transit': 0, 'affordable': 0, 'community': 0}
                   for hood in df['neighborhood']}

    investment_costs = {
        'transit': 50_000_000,      # $50M per unit
        'affordable': 30_000_000,   # $30M per unit
        'community': 10_000_000     # $10M per unit
    }

    # Calculate baseline risk
    baseline_total_risk = df['baseline_displacement_risk'].sum()

    while remaining_budget > min(investment_costs.values()):
        best_roi = -float('inf')
        best_investment = None

        # Try each possible investment
        for hood in df['neighborhood']:
            for inv_type, cost in investment_costs.items():
                if cost > remaining_budget:
                    continue

                # Calculate ROI for this investment
                test_investments = investments.copy()
                test_investments[hood][inv_type] += cost

                # Simulate impact
                df_test = calculate_spillover_effects(df, test_investments, network)

                # Calculate improvement
                if target == 'minimize_displacement':
                    # ROI = reduction in total displacement risk per dollar
                    new_total_risk = df_test['baseline_displacement_risk'].sum() + \
                                   df_test['spillover_risk_impact'].sum()
                    risk_reduction = baseline_total_risk - new_total_risk
                    roi = risk_reduction / cost
                else:
                    # ROI = equity improvement (helping highest-risk neighborhoods)
                    high_risk_hoods = df[df['baseline_displacement_risk'] > 70]
                    roi = len(high_risk_hoods) / cost  # Simplified

                if roi > best_roi:
                    best_roi = roi
                    best_investment = (hood, inv_type, cost)

        if best_investment is None:
            break

        hood, inv_type, cost = best_investment
        investments[hood][inv_type] += cost
        remaining_budget -= cost

    return investments

def calculate_network_centrality(network):
    """
    Calculate which neighborhoods are most central/connected using PageRank-like algorithm.

    Args:
        network (dict): Neighborhood adjacency network

    Returns:
        dict: {neighborhood: centrality_score}
    """
    neighborhoods = list(network.keys())
    n = len(neighborhoods)

    # Initialize scores
    scores = {hood: 1.0 / n for hood in neighborhoods}

    # Iterative PageRank calculation
    damping = 0.85
    iterations = 20

    for _ in range(iterations):
        new_scores = {}
        for hood in neighborhoods:
            # Base score
            score = (1 - damping) / n

            # Add contributions from neighbors
            for other_hood in neighborhoods:
                if hood != other_hood and other_hood in network:
                    for neighbor, weight in network[other_hood]:
                        if neighbor == hood:
                            score += damping * scores[other_hood] * weight

            new_scores[hood] = score

        scores = new_scores

    return scores

if __name__ == "__main__":
    # Test the network effects
    print("Network effects module loaded successfully")
