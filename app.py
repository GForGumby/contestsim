import streamlit as st
import pandas as pd
import numpy as np
from numba import jit, prange
from st_paywall import add_auth
import io
from openpyxl import load_workbook

st.title("GUMBY SIMS- The first publicly available contest sim tool on Underdog. Take your game to the next level by simming out the highest ROI lineups,  optimal exposure percentages, and chalkiest field combos. ")

st.write("TO MANAGE YOUR SUBSCRIPTION GO HERE https://billing.stripe.com/p/login/9AQ4jFeRDbHTaIMfYY")
st.write("Please watch the entirety of this video explaining the product before purchasing (https://www.youtube.com/watch?v=tF9BU7yNdDI). I am available on twitter @GumbyUD for any questions or concerns about the product.")
st.write("On a CFB sized slate, it takes about 3 mins to run the draft sim (1884 sims), and about 5 mins to run 5000 instances of the projection sim.")
st.write("For the team stacking bonus, use .99 if you want stack frequency to mimic real drafts. Use .98 if you want slightly more, and 1.00 for no stacking. The lower it is the more frequent stacks are in your field of lineups. I wouldn't make it lower than .95 except for MLB, which I am currently still testing.") 
st.write("Current supported sports: NFL, CFB, PGA.")


add_auth(required=True)

# Path to your Excel file
sample_xlsx_path = 'Blank Analysis Template.xlsx'

# Read the Excel file
with open(sample_xlsx_path, 'rb') as file:
    excel_data = file.read()

# Create a download button
st.download_button(
    label="Blank Analysis Template",
    data=excel_data,
    file_name='Blank Analysis Template.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
)

st.write("Paste your sim results and draft results into the above file for more automated analysis")








st.subheader("NFL BR WEEK 1")


# Function to simulate a single draft
def simulate_draft(df, starting_team_num, num_teams=6, num_rounds=6, team_bonus=.99):
    df_copy = df.copy()
    df_copy['Simulated ADP'] = np.random.normal(df_copy['adp'], df_copy['adpsd'])
    df_copy.sort_values('Simulated ADP', inplace=True)
    
    teams = {f'Team {i + starting_team_num}': [] for i in range(num_teams)}
    team_positions = {f'Team {i + starting_team_num}': {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0} for i in range(num_teams)}
    teams_stack = {f'Team {i + starting_team_num}': [] for i in range(num_teams)}
    
    for round_num in range(num_rounds):
        draft_order = list(range(num_teams)) if round_num % 2 == 0 else list(range(num_teams))[::-1]
        for pick_num in draft_order:
            if not df_copy.empty:
                team_name = f'Team {pick_num + starting_team_num}'
                
                draftable_positions = []
                if team_positions[team_name]["QB"] < 1:
                    draftable_positions.append("QB")
                if team_positions[team_name]["RB"] < 1:
                    draftable_positions.append("RB")
                if team_positions[team_name]["WR"] < 2:
                    draftable_positions.append("WR")
                if team_positions[team_name]["TE"] < 1:
                    draftable_positions.append("TE")
                if team_positions[team_name]["FLEX"] < 1 and (team_positions[team_name]["RB"] + team_positions[team_name]["WR"] < 5):
                    draftable_positions.append("FLEX")
                
                df_filtered = df_copy.loc[
                    df_copy['position'].isin(draftable_positions) | 
                    ((df_copy['position'].isin(['RB', 'WR'])) & ('FLEX' in draftable_positions))
                ].copy()
                
                if df_filtered.empty:
                    continue
                
                df_filtered['Adjusted ADP'] = df_filtered.apply(
                    lambda x: x['Simulated ADP'] * team_bonus 
                    if x['team'] in teams_stack[team_name] else x['Simulated ADP'],
                    axis=1
                )
                
                df_filtered.sort_values('Adjusted ADP', inplace=True)
                
                selected_player = df_filtered.iloc[0]
                teams[team_name].append(selected_player)
                teams_stack[team_name].append(selected_player['team'])
                position = selected_player['position']
                if position in ["RB", "WR"]:
                    if team_positions[team_name][position] < {"RB": 1, "WR": 2}[position]:
                        team_positions[team_name][position] += 1
                    else:
                        team_positions[team_name]["FLEX"] += 1
                else:
                    team_positions[team_name][position] += 1
                df_copy = df_copy.loc[df_copy['player_id'] != selected_player['player_id']]
    
    return teams

def run_simulations(df, num_simulations=10, num_teams=6, num_rounds=6, team_bonus=.99):
    all_drafts = []
    for sim_num in range(num_simulations):
        starting_team_num = sim_num * num_teams + 1
        draft_result = simulate_draft(df, starting_team_num, num_teams, num_rounds, team_bonus)
        all_drafts.append(draft_result)
    return all_drafts

@jit(nopython=True)
def generate_projection(median, std_dev):
    fluctuation = np.random.uniform(-0.01, 0.01) * median
    return max(0, np.random.normal(median, std_dev) + fluctuation)

@jit(nopython=True)
def get_payout(rank):
    if rank == 1:
        return 5000.00
    elif rank == 2:
        return 2500.00
    elif rank == 3:
        return 1250.00
    elif rank == 4:
        return 750.00
    elif rank == 5:
        return 600.00
    elif 6 <= rank <= 10:
        return 500.00
    elif 11 <= rank <= 15:
        return 400.00
    elif 16 <= rank <= 20:
        return 300.00
    elif 21 <= rank <= 25:
        return 250.00
    elif 26 <= rank <= 30:
        return 200.00
    elif 31 <= rank <= 35:
        return 150.00
    elif 36 <= rank <= 40:
        return 100.00
    elif 41 <= rank <= 45:
        return 75.00
    elif 46 <= rank <= 50:
        return 60.00
    elif 51 <= rank <= 55:
        return 50.00
    elif 56 <= rank <= 85:
        return 40.00
    elif 86 <= rank <= 135:
        return 30.00
    elif 136 <= rank <= 210:
        return 25.00
    elif 211 <= rank <= 325:
        return 20.00
    elif 326 <= rank <= 505:
        return 15.00
    elif 506 <= rank <= 2495:
        return 10.00
    else:
        return 0.00

def prepare_draft_results(draft_results_df):
    teams = draft_results_df['Team'].unique()
    num_teams = len(teams)
    draft_results = np.empty((num_teams, 6), dtype='U50')

    for idx, team in enumerate(teams):
        team_players = draft_results_df[draft_results_df['Team'] == team]
        for i in range(1, 7):
            player_col = f'Player_{i}_Name'
            if player_col in team_players.columns and i <= len(team_players):
                draft_results[idx, i - 1] = team_players.iloc[0][player_col]
            else:
                draft_results[idx, i - 1] = "N/A"  # Placeholder for missing players

    return draft_results, teams

def simulate_team_projections(draft_results, projection_lookup, num_simulations):
    num_teams = draft_results.shape[0]
    total_payouts = np.zeros(num_teams)

    for sim in range(num_simulations):
        total_points = np.zeros(num_teams)
        for i in range(num_teams):
            for j in range(6):  # Loop through all 6 players
                player_name = draft_results[i, j]
                if player_name in projection_lookup:
                    proj, projsd = projection_lookup[player_name]
                    simulated_points = generate_projection(proj, projsd)
                    total_points[i] += simulated_points

        ranks = total_points.argsort()[::-1].argsort() + 1
        payouts = np.array([get_payout(rank) for rank in ranks])
        total_payouts += payouts

    avg_payouts = total_payouts / num_simulations
    return avg_payouts

def run_parallel_simulations(num_simulations, draft_results_df, projection_lookup):
    draft_results, teams = prepare_draft_results(draft_results_df)
    
    all_players = [player for team in draft_results for player in team if player != 'N/A']
    filtered_projection_lookup = {player: projection_lookup[player] for player in all_players if player in projection_lookup}
    
    avg_payouts = simulate_team_projections(draft_results, filtered_projection_lookup, num_simulations)
    
    final_results = pd.DataFrame({
        'Team': teams,
        'Average_Payout': avg_payouts
    })
    
    return final_results

# Streamlit app
sample_csv_path = 'NFL WEEK 1 ADP TEMPLATE.csv'
with open(sample_csv_path, 'rb') as file:
    sample_csv = file.read()

st.download_button(
    label="NFL WEEK 1 ADP TEMPLATE",
    data=sample_csv,
    file_name='NFL WEEK 1 ADP TEMPLATE.csv',
    mime='text/csv',
)

# File upload for ADP
adp_file = st.file_uploader("Upload your NFL ADP CSV file", type=["csv"])

if adp_file is not None:
    df = pd.read_csv(adp_file)
    if 'player_id' not in df.columns:
        df['player_id'] = df.index
    
    st.write("ADP Data Preview:")
    st.dataframe(df.head())
    
    num_simulations = st.number_input("Number of simulations", min_value=1, value=10)
    num_teams = st.number_input("Number of teams", min_value=2, value=6)
    num_rounds = st.number_input("Number of rounds", min_value=1, value=6)
    team_bonus = st.number_input("Team stacking bonus", min_value=0.0, value=0.99)
    
    if st.button("Run Draft Simulation"):
        all_drafts = run_simulations(df, num_simulations, num_teams, num_rounds, team_bonus)
        draft_results = []
        for sim_num, draft in enumerate(all_drafts):
            for team, players in draft.items():
                result_entry = {
                    'Simulation': sim_num + 1,
                    'Team': team,
                }
                for i, player in enumerate(players):
                    result_entry.update({
                        f'Player_{i+1}_Name': player['name'],
                        f'Player_{i+1}_Position': player['position'],
                        f'Player_{i+1}_Team': player['team']
                    })
                draft_results.append(result_entry)
        
        draft_results_df = pd.DataFrame(draft_results)
        st.write("Draft Simulation Results:")
        st.dataframe(draft_results_df)
        
        csv = draft_results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Draft Results",
            data=csv,
            file_name='nfl_draft_results.csv',
            mime='text/csv',
        )

# File uploaders for projections and draft results
projections_file = st.file_uploader("Choose a CSV with NFL player projections", type="csv")
draft_results_file = st.file_uploader("Choose a CSV file with NFL draft results", type="csv")

if projections_file is not None and draft_results_file is not None:
    projections_df = pd.read_csv(projections_file)
    draft_results_df = pd.read_csv(draft_results_file)
    
    st.write("Projections and draft results loaded successfully!")
    
    st.subheader("Sample of loaded projections:")
    st.write(projections_df.head())
    
    st.subheader("Sample of loaded draft results:")
    st.write(draft_results_df.head())

    projection_lookup = dict(zip(projections_df['name'], zip(projections_df['proj'], projections_df['projsd'])))

    num_simulations = st.number_input("Number of simulations to run", min_value=100, max_value=100000, value=10000, step=100)

    if st.button("Run Projection Simulations"):
        with st.spinner('Running simulations...'):
            final_results = run_parallel_simulations(num_simulations, draft_results_df, projection_lookup)

        st.subheader("Projection Simulation Results:")
        st.write(final_results)

        csv = final_results.to_csv(index=False)
        st.download_button(
            label="Download Projection Results as CSV",
            data=csv,
            file_name="projection_simulation_results.csv",
            mime="text/csv",
        )



st.subheader("------------------------------------------------------------------------------")

st.subheader("COLLEGE FOOTBALL WEEK 1 BR")


# Function to simulate a single draft
def simulate_draft(df, starting_team_num, num_teams=6, num_rounds=6, team_bonus=.99):
    df_copy = df.copy()
    df_copy['Simulated ADP'] = np.random.normal(df_copy['adp'], df_copy['adpsd'])
    df_copy.sort_values('Simulated ADP', inplace=True)
    
    teams = {f'Team {i + starting_team_num}': [] for i in range(num_teams)}
    team_positions = {f'Team {i + starting_team_num}': {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0} for i in range(num_teams)}
    teams_stack = {f'Team {i + starting_team_num}': [] for i in range(num_teams)}
    
    for round_num in range(num_rounds):
        draft_order = list(range(num_teams)) if round_num % 2 == 0 else list(range(num_teams))[::-1]
        for pick_num in draft_order:
            if not df_copy.empty:
                team_name = f'Team {pick_num + starting_team_num}'
                
                draftable_positions = []
                if team_positions[team_name]["QB"] < 1:
                    draftable_positions.append("QB")
                if team_positions[team_name]["RB"] < 1:
                    draftable_positions.append("RB")
                if team_positions[team_name]["WR"] < 2:
                    draftable_positions.append("WR")
                if team_positions[team_name]["TE"] < 1:
                    draftable_positions.append("TE")
                if team_positions[team_name]["FLEX"] < 1 and (team_positions[team_name]["RB"] + team_positions[team_name]["WR"] < 5):
                    draftable_positions.append("FLEX")
                
                df_filtered = df_copy.loc[
                    df_copy['position'].isin(draftable_positions) | 
                    ((df_copy['position'].isin(['RB', 'WR'])) & ('FLEX' in draftable_positions))
                ].copy()
                
                if df_filtered.empty:
                    continue
                
                df_filtered['Adjusted ADP'] = df_filtered.apply(
                    lambda x: x['Simulated ADP'] * team_bonus 
                    if x['team'] in teams_stack[team_name] else x['Simulated ADP'],
                    axis=1
                )
                
                df_filtered.sort_values('Adjusted ADP', inplace=True)
                
                selected_player = df_filtered.iloc[0]
                teams[team_name].append(selected_player)
                teams_stack[team_name].append(selected_player['team'])
                position = selected_player['position']
                if position in ["RB", "WR"]:
                    if team_positions[team_name][position] < {"RB": 1, "WR": 2}[position]:
                        team_positions[team_name][position] += 1
                    else:
                        team_positions[team_name]["FLEX"] += 1
                else:
                    team_positions[team_name][position] += 1
                df_copy = df_copy.loc[df_copy['player_id'] != selected_player['player_id']]
    
    return teams

def run_simulations(df, num_simulations=10, num_teams=6, num_rounds=6, team_bonus=.99):
    all_drafts = []
    for sim_num in range(num_simulations):
        starting_team_num = sim_num * num_teams + 1
        draft_result = simulate_draft(df, starting_team_num, num_teams, num_rounds, team_bonus)
        all_drafts.append(draft_result)
    return all_drafts

@jit(nopython=True)
def generate_projection(median, std_dev):
    fluctuation = np.random.uniform(-0.01, 0.01) * median
    return max(0, np.random.normal(median, std_dev) + fluctuation)

@jit(nopython=True)
def get_payout(rank):
    if rank == 1:
        return 5000.00
    elif rank == 2:
        return 2500.00
    elif rank == 3:
        return 1250.00
    elif rank == 4:
        return 750.00
    elif rank == 5:
        return 600.00
    elif 6 <= rank <= 10:
        return 500.00
    elif 11 <= rank <= 15:
        return 400.00
    elif 16 <= rank <= 20:
        return 300.00
    elif 21 <= rank <= 25:
        return 250.00
    elif 26 <= rank <= 30:
        return 200.00
    elif 31 <= rank <= 35:
        return 150.00
    elif 36 <= rank <= 40:
        return 100.00
    elif 41 <= rank <= 45:
        return 75.00
    elif 46 <= rank <= 50:
        return 60.00
    elif 51 <= rank <= 55:
        return 50.00
    elif 56 <= rank <= 85:
        return 40.00
    elif 86 <= rank <= 135:
        return 30.00
    elif 136 <= rank <= 210:
        return 25.00
    elif 211 <= rank <= 325:
        return 20.00
    elif 326 <= rank <= 505:
        return 15.00
    elif 506 <= rank <= 2495:
        return 10.00
    else:
        return 0.00

def prepare_draft_results(draft_results_df):
    teams = draft_results_df['Team'].unique()
    num_teams = len(teams)
    draft_results = np.empty((num_teams, 6), dtype='U50')

    for idx, team in enumerate(teams):
        team_players = draft_results_df[draft_results_df['Team'] == team]
        for i in range(1, 7):
            player_col = f'Player_{i}_Name'
            if player_col in team_players.columns and i <= len(team_players):
                draft_results[idx, i - 1] = team_players.iloc[0][player_col]
            else:
                draft_results[idx, i - 1] = "N/A"  # Placeholder for missing players

    return draft_results, teams

def simulate_team_projections(draft_results, projection_lookup, num_simulations):
    num_teams = draft_results.shape[0]
    total_payouts = np.zeros(num_teams)

    for sim in range(num_simulations):
        total_points = np.zeros(num_teams)
        for i in range(num_teams):
            for j in range(6):  # Loop through all 6 players
                player_name = draft_results[i, j]
                if player_name in projection_lookup:
                    proj, projsd = projection_lookup[player_name]
                    simulated_points = generate_projection(proj, projsd)
                    total_points[i] += simulated_points

        ranks = total_points.argsort()[::-1].argsort() + 1
        payouts = np.array([get_payout(rank) for rank in ranks])
        total_payouts += payouts

    avg_payouts = total_payouts / num_simulations
    return avg_payouts

def run_parallel_simulations(num_simulations, draft_results_df, projection_lookup):
    draft_results, teams = prepare_draft_results(draft_results_df)
    
    all_players = [player for team in draft_results for player in team if player != 'N/A']
    filtered_projection_lookup = {player: projection_lookup[player] for player in all_players if player in projection_lookup}
    
    avg_payouts = simulate_team_projections(draft_results, filtered_projection_lookup, num_simulations)
    
    final_results = pd.DataFrame({
        'Team': teams,
        'Average_Payout': avg_payouts
    })
    
    return final_results

# Streamlit app
sample_csv_path = 'CFB WEEK 1 ADP TEMPLATE.csv'
with open(sample_csv_path, 'rb') as file:
    sample_csv = file.read()

st.download_button(
    label="CFB WEEK 1 ADP TEMPLATE",
    data=sample_csv,
    file_name='CFB WEEK 1 ADP TEMPLATE.csv',
    mime='text/csv',
    key="cfb_week1_adp_template_download"
)

# File upload for ADP
adp_file = st.file_uploader("Upload your ADP CSV file", type=["csv"])

if adp_file is not None:
    df = pd.read_csv(adp_file)
    if 'player_id' not in df.columns:
        df['player_id'] = df.index
    
    st.write("ADP Data Preview:")
    st.dataframe(df.head())
    
    num_simulations = st.number_input("Number of simulations", min_value=1, value=10)
    num_teams = st.number_input("Number of teams", min_value=2, value=6)
    num_rounds = st.number_input("Number of rounds", min_value=1, value=6)
    team_bonus = st.number_input("Team stacking bonus", min_value=0.0, value=0.99)
    
    if st.button("Run Draft Simulation"):
        all_drafts = run_simulations(df, num_simulations, num_teams, num_rounds, team_bonus)
        draft_results = []
        for sim_num, draft in enumerate(all_drafts):
            for team, players in draft.items():
                result_entry = {
                    'Simulation': sim_num + 1,
                    'Team': team,
                }
                for i, player in enumerate(players):
                    result_entry.update({
                        f'Player_{i+1}_Name': player['name'],
                        f'Player_{i+1}_Position': player['position'],
                        f'Player_{i+1}_Team': player['team']
                    })
                draft_results.append(result_entry)
        
        draft_results_df = pd.DataFrame(draft_results)
        st.write("Draft Simulation Results:")
        st.dataframe(draft_results_df)
        
        csv = draft_results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Draft Results",
            data=csv,
            file_name='draft_results.csv',
            mime='text/csv',
        )

# File uploaders for projections and draft results
projections_file = st.file_uploader("Choose a CSV file with player projections", type="csv")
draft_results_file = st.file_uploader("Choose a CSV file with draft results", type="csv")

if projections_file is not None and draft_results_file is not None:
    projections_df = pd.read_csv(projections_file)
    draft_results_df = pd.read_csv(draft_results_file)
    
    st.write("Projections and draft results loaded successfully!")
    
    st.subheader("Sample of loaded projections:")
    st.write(projections_df.head())
    
    st.subheader("Sample of loaded draft results:")
    st.write(draft_results_df.head())

    projection_lookup = dict(zip(projections_df['name'], zip(projections_df['proj'], projections_df['projsd'])))

    num_simulations = st.number_input("Number of simulations to run", min_value=100, max_value=100000, value=10000, step=100)

    if st.button("Run Projection Simulations"):
        with st.spinner('Running simulations...'):
            final_results = run_parallel_simulations(num_simulations, draft_results_df, projection_lookup)

        st.subheader("Projection Simulation Results:")
        st.write(final_results)

        csv = final_results.to_csv(index=False)
        st.download_button(
            label="Download Projection Results as CSV",
            data=csv,
            file_name="projection_simulation_results.csv",
            mime="text/csv",
        )















st.subheader("------------------------------------------------------------------------------")





st.subheader("PGA Tour Championship Rd2")


def simulate_draft(df, starting_team_num, num_teams=6, num_rounds=6, team_bonus=1.00):
    df_copy = df.copy()
    df_copy['Simulated ADP'] = np.random.normal(df_copy['adp'], df_copy['adpsd'])
    df_copy.sort_values('Simulated ADP', inplace=True)
    
    teams = {f'Team {i + starting_team_num}': [] for i in range(num_teams)}
    teams_stack = {f'Team {i + starting_team_num}': [] for i in range(num_teams)}
    
    for round_num in range(num_rounds):
        draft_order = list(range(num_teams)) if round_num % 2 == 0 else list(range(num_teams))[::-1]
        for pick_num in draft_order:
            if not df_copy.empty:
                team_name = f'Team {pick_num + starting_team_num}'
                
                df_filtered = df_copy.copy()
                
                df_filtered['Adjusted ADP'] = df_filtered.apply(
                    lambda x: x['Simulated ADP'] * team_bonus 
                    if x['team'] in teams_stack[team_name] else x['Simulated ADP'],
                    axis=1
                )
                
                df_filtered.sort_values('Adjusted ADP', inplace=True)
                
                selected_player = df_filtered.iloc[0]
                selected_player['position'] = 'G'  # Set all positions to 'G'
                teams[team_name].append(selected_player)
                teams_stack[team_name].append(selected_player['team'])
                df_copy = df_copy.loc[df_copy['player_id'] != selected_player['player_id']]
    
    return teams

def run_simulations(df, num_simulations=10, num_teams=6, num_rounds=6, team_bonus=1.00):
    all_drafts = []
    for sim_num in range(num_simulations):
        starting_team_num = sim_num * num_teams + 1
        draft_result = simulate_draft(df, starting_team_num, num_teams, num_rounds, team_bonus)
        all_drafts.append(draft_result)
    return all_drafts

@jit(nopython=True)
def generate_projection(median, std_dev):
    fluctuation = np.random.uniform(-0.01, 0.01) * median
    return max(0, np.random.normal(median, std_dev) + fluctuation)

@jit(nopython=True)
def get_payout(rank):
    if rank == 1:
        return 2500.00
    elif rank == 2:
        return 1750.00
    elif rank == 3:
        return 1000.00
    elif rank == 4:
        return 825.00
    elif rank == 5:
        return 600.00
    elif 6 <= rank <= 7:
        return 500.00
    elif 8 <= rank <= 10:
        return 400.00
    elif 11 <= rank <= 13:
        return 300.00
    elif 14 <= rank <= 16:
        return 200.00
    elif 17 <= rank <= 20:
        return 150.00
    elif 21 <= rank <= 25:
        return 100.00
    elif 26 <= rank <= 30:
        return 90.00
    elif 31 <= rank <= 35:
        return 80.00
    elif 36 <= rank <= 40:
        return 75.00
    elif 41 <= rank <= 45:
        return 60.00
    elif 46 <= rank <= 50:
        return 50.00
    elif 51 <= rank <= 100:
        return 25.00
    elif 101 <= rank <= 400:
        return 15.00
    elif 401 <= rank <= 1000:
        return 10.00
    else:
        return 0.00  # No payout for ranks beyond 1000

def prepare_draft_results(draft_results_df):
    teams = draft_results_df['Team'].unique()
    num_teams = len(teams)
    draft_results = np.empty((num_teams, 6), dtype='U50')

    for idx, team in enumerate(teams):
        team_players = draft_results_df[draft_results_df['Team'] == team]
        for i in range(1, 7):
            player_col = f'Player_{i}_Name'
            if player_col in team_players.columns and i <= len(team_players):
                draft_results[idx, i - 1] = team_players.iloc[0][player_col]
            else:
                draft_results[idx, i - 1] = "N/A"  # Placeholder for missing players

    return draft_results, teams

def simulate_team_projections(draft_results, projection_lookup, num_simulations):
    num_teams = draft_results.shape[0]
    total_payouts = np.zeros(num_teams)

    for sim in range(num_simulations):
        total_points = np.zeros(num_teams)
        for i in range(num_teams):
            for j in range(6):  # Loop through all 6 players
                player_name = draft_results[i, j]
                if player_name in projection_lookup:
                    proj, projsd = projection_lookup[player_name]
                    simulated_points = generate_projection(proj, projsd)
                    total_points[i] += simulated_points

        ranks = total_points.argsort()[::-1].argsort() + 1
        payouts = np.array([get_payout(rank) for rank in ranks])
        total_payouts += payouts

    avg_payouts = total_payouts / num_simulations
    return avg_payouts

def run_parallel_simulations(num_simulations, draft_results_df, projection_lookup):
    draft_results, teams = prepare_draft_results(draft_results_df)
    
    all_players = [player for team in draft_results for player in team if player != 'N/A']
    filtered_projection_lookup = {player: projection_lookup[player] for player in all_players if player in projection_lookup}
    
    avg_payouts = simulate_team_projections(draft_results, filtered_projection_lookup, num_simulations)
    
    final_results = pd.DataFrame({
        'Team': teams,
        'Average_Payout': avg_payouts
    })
    
    return final_results

# Streamlit app
sample_csv_path = 'PGA TOUR CHAMP RD2.csv'
with open(sample_csv_path, 'rb') as file:
    sample_csv = file.read()

st.download_button(
    label="PGA TOUR CHAMP RD2",
    data=sample_csv,
    file_name='PGA TOUR CHAMP RD2.csv',
    mime='text/csv',
)

# File upload for ADP
adp_file = st.file_uploader("Upload your PGA ADP CSV file", type=["csv"])

if adp_file is not None:
    df = pd.read_csv(adp_file)
    if 'player_id' not in df.columns:
        df['player_id'] = df.index
    
    st.write("ADP Data Preview:")
    st.dataframe(df.head())
    
    num_simulations = st.number_input("Number of simulations", min_value=1, value=10)
    num_teams = st.number_input("Number of teams", min_value=2, value=6)
    num_rounds = st.number_input("Number of rounds", min_value=1, value=6)
    team_bonus = st.number_input("Team stacking bonus", min_value=0.0, value=0.99)
    
    if st.button("Run Draft Simulation"):
        all_drafts = run_simulations(df, num_simulations, num_teams, num_rounds, team_bonus)
        draft_results = []
        for sim_num, draft in enumerate(all_drafts):
            for team, players in draft.items():
                result_entry = {
                    'Simulation': sim_num + 1,
                    'Team': team,
                }
                for i, player in enumerate(players):
                    result_entry.update({
                        f'Player_{i+1}_Name': player['name'],
                        f'Player_{i+1}_Position': player['position'],
                        f'Player_{i+1}_Team': player['team']
                    })
                draft_results.append(result_entry)
        
        draft_results_df = pd.DataFrame(draft_results)
        st.write("Draft Simulation Results:")
        st.dataframe(draft_results_df)
        
        csv = draft_results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Draft Results",
            data=csv,
            file_name='draft_results.csv',
            mime='text/csv',
        )

# File uploaders for projections and draft results
projections_file = st.file_uploader("Choose a CSV file with PGA projections", type="csv")
draft_results_file = st.file_uploader("Choose a CSV file with PGA draft results", type="csv")

if projections_file is not None and draft_results_file is not None:
    projections_df = pd.read_csv(projections_file)
    draft_results_df = pd.read_csv(draft_results_file)
    
    st.write("Projections and draft results loaded successfully!")
    
    st.subheader("Sample of loaded projections:")
    st.write(projections_df.head())
    
    st.subheader("Sample of loaded draft results:")
    st.write(draft_results_df.head())

    projection_lookup = dict(zip(projections_df['name'], zip(projections_df['proj'], projections_df['projsd'])))

    num_simulations = st.number_input("Number of simulations to run", min_value=100, max_value=100000, value=10000, step=100)

    if st.button("Run Projection Simulations"):
        with st.spinner('Running simulations...'):
            final_results = run_parallel_simulations(num_simulations, draft_results_df, projection_lookup)

        st.subheader("Projection Simulation Results:")
        st.write(final_results)

        csv = final_results.to_csv(index=False)
        st.download_button(
            label="Download PGA Projection Results as CSV",
            data=csv,
            file_name="pga_projection_simulation_results.csv",
            mime="text/csv",
        )

