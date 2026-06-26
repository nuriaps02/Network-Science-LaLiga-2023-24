"""
utils.py

Shared variables, data loaders, and metric functions for the TFM analysis.

"""

import json
import zipfile
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats


# Data paths

zip_path       = 'data/laliga.zip'
folder_laliga  = "laliga_2023_24"


# Data loading 

def stream_matches_from_zip(zip_path, folder_in_zip, file_type):
    """Streams match event files directly from a ZIP to save memory."""
    with zipfile.ZipFile(zip_path, "r") as z:
        for file_name in z.namelist():
            if file_name.startswith(folder_in_zip) and file_name.endswith(file_type):
                with z.open(file_name) as f:
                    match_id = file_name.split('/')[-1].split('_')[0]
                    yield match_id, json.load(f)

def load_all_teams(zip_path, folder, n=20):
    """Returns a dict {team_id: team_name} for the first n teams found."""
    all_teams = {}
    for _, events in stream_matches_from_zip(zip_path, folder, "_events.json"):
        for e in events:
            if 'team' in e:
                tid   = e['team']['id']
                tname = e['team']['name']
                if tid not in all_teams:
                    all_teams[tid] = tname
        if len(all_teams) >= n:
            break
    return all_teams


# Switching Factor 

def calculate_match_sf(events, team_id):
    """
    Global SF for a team in a match:
        <S>_i = 0.5 * (P_in* + P_out*) / P_passes

    P_in*  = Ball Recoveries + successful Interceptions
    P_out* = incomplete Passes + Dispossessed + Miscontrol
    """
    p_in_star  = 0
    p_out_star = 0
    p_passes   = 0

    for e in events:
        if 'team' not in e or e['team']['id'] != team_id:
            continue
        etype = e['type']['name']

        if etype == 'Pass':
            p_passes += 1
            if 'outcome' in e['pass']:
                p_out_star += 1
        elif etype == 'Ball Recovery':
            p_in_star += 1
        elif etype == 'Interception':
            outcome = e.get('interception', {}).get('outcome', {}).get('name')
            if outcome in ['Success', 'Won', 'Success In Play']:
                p_in_star += 1
        elif etype in ['Dispossessed', 'Miscontrol']:
            p_out_star += 1

    return 0.5 * ((p_in_star + p_out_star) / p_passes) if p_passes > 0 else 0


# Match result 

def get_match_result(events, team_id):
    """
    Returns (goals_for, goals_against, result_str, points).
    result_str is one of: 'win', 'draw', 'lost'.
    """
    teams_in_match = {e['team']['id'] for e in events if 'team' in e}
    goals_for      = 0
    goals_against  = 0

    for e in events:
        if (e['type']['name'] == 'Shot' and
                e.get('shot', {}).get('outcome', {}).get('name') == 'Goal'):
            if e['team']['id'] == team_id:
                goals_for += 1
            elif e['team']['id'] in teams_in_match:
                goals_against += 1

    res = ('win'  if goals_for > goals_against else
           'lost' if goals_for < goals_against else 'draw')
    pts = {'win': 3, 'draw': 1, 'lost': 0}[res]
    return goals_for, goals_against, res, pts


# Eigenvector Centrality

def calcular_ec_equip(events, team_name):
    """
    Builds a weighted directed passing network (successful passes only: outcome is None in StatsBomb = success)
    and returns:
        avg_ec  — mean eigenvector centrality across all players
        ec_dict — {player_name: ec_value}
    Returns (np.nan, {}) if the graph is too sparse to converge.
    """
    passes = [
        e for e in events
        if e.get('type', {}).get('name') == 'Pass'
        and e.get('team', {}).get('name') == team_name
        and e.get('pass', {}).get('outcome') is None  # None = success in StatsBomb
    ]
    if len(passes) < 5:
        return np.nan, {}

    G = nx.DiGraph()
    for p in passes:
        u = p.get('player', {}).get('name')
        v = p.get('pass', {}).get('recipient', {}).get('name')
        if u and v:
            if G.has_edge(u, v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1)

    if G.number_of_nodes() < 3:
        return np.nan, {}

    try:
        ec_dict = nx.eigenvector_centrality(G, weight='weight', max_iter=1000, tol=1e-6)
        avg_ec  = np.mean(list(ec_dict.values()))
        return avg_ec, ec_dict
    except nx.PowerIterationFailedConvergence:
        return np.nan, {}


# Season-level aggregation

def analisi_sf_ec_temporada(team_id, team_name, zip_path, folder):
    """
    Iterates every match for a given team and computes SF, EC, result
    and points per match.

    Returns a DataFrame with columns:
        match_id | sf | ec | result | points
    Rows with NaN EC (sparse graphs) are dropped.
    """
    results = []
    for m_id, events in stream_matches_from_zip(zip_path, folder, "_events.json"):
        teams_in_match = {e['team']['id'] for e in events if 'team' in e}
        if team_id not in teams_in_match:
            continue

        sf             = calculate_match_sf(events, team_id)
        avg_ec, _      = calcular_ec_equip(events, team_name)
        _, _, res, pts = get_match_result(events, team_id)

        results.append({
            'match_id': m_id,
            'sf':       sf,
            'ec':       avg_ec,
            'result':   res,
            'points':   pts,
        })

    return pd.DataFrame(results).dropna()


# Build passing networks

def build_passing_network(events, team_name):
    """
    Build a directed weighted passing network for a team in a single match.
    Nodes = players, edges = completed passes (weight = number of passes).
    """
    passes = [
        e for e in events
        if e.get('type', {}).get('name') == 'Pass'
        and e.get('team', {}).get('name') == team_name
        and e.get('pass', {}).get('outcome') is None
        and e.get('player', {}).get('name')
        and e.get('pass', {}).get('recipient', {}).get('name')
    ]
    
    G = nx.DiGraph()
    for p in passes:
        passer    = p['player']['name']
        recipient = p['pass']['recipient']['name']
        if G.has_edge(passer, recipient):
            G[passer][recipient]['weight'] += 1
        else:
            G.add_edge(passer, recipient, weight=1)
    
    return G

def build_consolidated_network(team_id, team_name, zip_path, folder):
    """
    Build a single passing network aggregating all matches of a team in the season.
    Edge weights = total number of completed passes between two players across the season.
    """
    G_consolidated = nx.DiGraph()
    n_matches = 0
    
    for m_id, events in stream_matches_from_zip(zip_path, folder, "_events.json"):
        teams_in_match = {e['team']['id'] for e in events if 'team' in e}
        if team_id not in teams_in_match:
            continue
        
        G_match = build_passing_network(events, team_name)
        
        # Sum edge weights across matches
        for u, v, d in G_match.edges(data=True):
            if G_consolidated.has_edge(u, v):
                G_consolidated[u][v]['weight'] += d['weight']
            else:
                G_consolidated.add_edge(u, v, weight=d['weight'])
        
        n_matches += 1
    
    return G_consolidated, n_matches


# Topological metrics

def compute_topological_metrics(G):
    """
    Compute global topological metrics for a passing network.
    Returns a dict with all metrics.
    """
    if G.number_of_nodes() == 0:
        return {
            'n_nodes': 0, 'n_edges': 0, 'density': np.nan,
            'efficiency': np.nan, 'clustering': np.nan,
            'avg_path_length': np.nan, 'reciprocity': np.nan
        }
    
    metrics = {
        'n_nodes':  G.number_of_nodes(),
        'n_edges':  G.number_of_edges(),
        'density':  nx.density(G),
    }
    
    # Global efficiency (uses reciprocal of shortest path)
    try:
        metrics['efficiency'] = nx.global_efficiency(G.to_undirected())
    except Exception:
        metrics['efficiency'] = np.nan
    
    # Average clustering coefficient (weighted, undirected version)
    try:
        metrics['clustering'] = nx.average_clustering(G.to_undirected(), weight='weight')
    except Exception:
        metrics['clustering'] = np.nan
    
    # Average shortest path length (only on largest connected component)
    try:
        if nx.is_strongly_connected(G):
            metrics['avg_path_length'] = nx.average_shortest_path_length(G)
        else:
            largest_cc = max(nx.weakly_connected_components(G), key=len)
            G_sub = G.subgraph(largest_cc)
            if nx.is_strongly_connected(G_sub):
                metrics['avg_path_length'] = nx.average_shortest_path_length(G_sub)
            else:
                metrics['avg_path_length'] = nx.average_shortest_path_length(G_sub.to_undirected())
    except Exception:
        metrics['avg_path_length'] = np.nan
    
    # Reciprocity (fraction of mutual edges in directed network)
    try:
        metrics['reciprocity'] = nx.reciprocity(G)
    except Exception:
        metrics['reciprocity'] = np.nan
    
    return metrics


# Player centralities

def compute_player_centralities(G):
    """
    Compute PageRank and betweenness centrality for all players.
    Returns DataFrame ordered by PageRank descending.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame()
    
    try:
        pr = nx.pagerank(G, weight='weight')
    except nx.PowerIterationFailedConvergence:
        return pd.DataFrame()
    
    bt = nx.betweenness_centrality(G, weight='weight')
    
    df = pd.DataFrame([
        {'player': player, 'pagerank': pr[player], 'betweenness': bt[player]}
        for player in pr
    ])
    df = df.sort_values('pagerank', ascending=False).reset_index(drop=True)
    df['pagerank_pct'] = (df['pagerank'] * 100).round(2)
    return df

def compute_pagerank_match(G):
    """Compute PageRank for all players in the passing network."""
    if G.number_of_nodes() == 0:
        return {}
    try:
        return nx.pagerank(G, weight='weight')
    except nx.PowerIterationFailedConvergence:
        return {}

def compute_betweenness_match(G):
    """Compute betweenness centrality. Invert weights (more passes = shorter distance)."""
    if G.number_of_nodes() == 0:
        return {}
    G_inv = G.copy()
    for u, v, d in G_inv.edges(data=True):
        d['weight'] = 1.0 / d['weight']
    return nx.betweenness_centrality(G_inv, weight='weight')


# Robustness analysis

def random_attack(G, n_simulations=100, seed=42):
    """
    Remove players at random as a control. Average over n_simulations.
    Returns a list of mean efficiency values after each removal.
    """
    if G.number_of_nodes() == 0:
        return []
    
    rng = np.random.default_rng(seed)
    n_nodes = G.number_of_nodes()
    all_curves = []
    
    for sim in range(n_simulations):
        G_work = G.copy()
        eff_curve = [nx.global_efficiency(G_work.to_undirected())]
        
        nodes_order = list(G_work.nodes())
        rng.shuffle(nodes_order)
        
        for node in nodes_order[:-1]:   # leave the last one
            G_work.remove_node(node)
            if G_work.number_of_nodes() > 0:
                eff_curve.append(nx.global_efficiency(G_work.to_undirected()))
            else:
                eff_curve.append(0.0)
        
        all_curves.append(eff_curve)
    
    # Return the mean across simulations
    max_len = max(len(c) for c in all_curves)
    padded  = [c + [0.0] * (max_len - len(c)) for c in all_curves]
    return list(np.mean(padded, axis=0))

def compute_fragility(targeted_curve, random_curve):
    """
    Area between the random and targeted curves (after normalization).
    Both curves start at the same value (initial efficiency).
    The area difference measures how much faster the targeted attack collapses the network.
    """
    n = min(len(targeted_curve), len(random_curve))
    if n == 0:
        return np.nan
    
    targeted = np.array(targeted_curve[:n])
    random_  = np.array(random_curve[:n])
    
    # Normalize by initial efficiency to make values comparable across teams
    initial_eff = max(targeted[0], random_[0], 1e-10)
    targeted_n  = targeted / initial_eff
    random_n    = random_  / initial_eff
    
    # Area between curves
    area = np.trapz(random_n - targeted_n) / n
    return area


# Distribution descriptors

def gini_coefficient(values):
    """Compute Gini coefficient as a measure of inequality in PageRank distribution."""
    values = np.array(sorted(values))
    n = len(values)
    if n == 0 or sum(values) == 0:
        return 0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))

def distribution_descriptors(centrality_values):
    """
    Entropy / effective-number descriptors of a centrality distribution.
    Input: 1D array-like of non-negative centrality values (one per player).
    """
    c = np.asarray(centrality_values, dtype=float)
    c = c[c > 0]                      # drop zero-centrality nodes (do not contribute to p ln p)
    N = len(c)
    if N == 0 or c.sum() == 0:
        return {k: np.nan for k in
                ['N', 'H_nats', 'S_norm', 'N_eff', 'N_eff_frac', 'gini']}

    p = c / c.sum()                   # normalize to a probability distribution

    H     = -np.sum(p * np.log(p))    # Shannon entropy (nats)
    S     = H / np.log(N) if N > 1 else 0.0
    N_eff = 1.0 / np.sum(p ** 2)      # inverse Simpson (Hill number, order 2)

    return {
        'N':          N,
        'H_nats':     H,
        'S_norm':     S,
        'N_eff':      N_eff,
        'N_eff_frac': N_eff / N,
        'gini':       gini_coefficient(c),
    }

def entropy_neff(centrality_values):
    """Tuple wrapper around distribution_descriptors."""
    d = distribution_descriptors(centrality_values)
    return d['N'], d['S_norm'], d['N_eff'], d['N_eff_frac']


# We load all_teams at import time

all_teams = load_all_teams(zip_path, folder_laliga)
