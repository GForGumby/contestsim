import streamlit as st
import pandas as pd
import numpy as np
from numba import jit, prange
from st_paywall import add_auth

import streamlit as st


st.title("Projections")
import pandas as pd
import numpy as np
import streamlit as st

# Add paywall
add_auth(required=True)

# Function to simulate a single draft
def simulate_draft(df, starting_team_num, num_teams=6, num_rounds=6, team_bonus=.95):
    df_copy = df.copy()
    df_copy['Simulated ADP'] = np.random.normal(df_copy['adp'], df_copy['adpsd'])
    df_copy.sort_values('Simulated ADP', inplace=True)
    
    # Initialize the teams
    teams = {f'Team {i + starting_team_num}': [] for i in range(num_teams)}
    team_positions = {f'Team {i + starting_team_num}': {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0} for i in range(num_teams)}
    teams_stack = {f'Team {i + starting_team_num}': [] for i in range(num_teams)}
    
    # Snake draft order
    for round_num in range(num_rounds):
        draft_order = list(range(num_teams)) if round_num % 2 == 0 else list(range(num_teams))[::-1]
        for pick_num in draft_order:
            if not df_copy.empty:
                team_name = f'Team {pick_num + starting_team_num}'
                
                # Filter players based on positional requirements
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
                
                # Adjust Simulated ADP based on team stacking
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

# Function to run multiple simulations
def run_simulations(df, num_simulations=10, num_teams=6, num_rounds=6, team_bonus=.95):
    all_drafts = []

    for sim_num in range(num_simulations):
        starting_team_num = sim_num * num_teams + 1
        draft_result = simulate_draft(df, starting_team_num, num_teams, num_rounds, team_bonus)
        all_drafts.append(draft_result)
    
    return all_drafts

# Streamlit app
st.title('Week 0 Test BR Draft Sim')

# Download link for sample CSV
sample_csv_path = 'adp sheet test.csv'
with open(sample_csv_path, 'rb') as file:
    sample_csv = file.read()

st.download_button(
    label="Download sample CSV",
    data=sample_csv,
    file_name='adp_sheet_test.csv',
    mime='text/csv',
)

# File upload
uploaded_file = st.file_uploader("Upload your ADP CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Check if player_id exists, if not, create it
    if 'player_id' not in df.columns:
        df['player_id'] = df.index
    
    st.write("Data Preview:")
    st.dataframe(df.head())
    
    # Parameters for the simulation
    num_simulations = st.number_input("Number of simulations", min_value=1, value=10)
    num_teams = st.number_input("Number of teams", min_value=2, value=6)
    num_rounds = st.number_input("Number of rounds", min_value=1, value=6)
    team_bonus = st.number_input("Team stacking bonus", min_value=0.0, value=0.95)

    
    if st.button("Run Simulation"):
        all_drafts = run_simulations(df, num_simulations, num_teams, num_rounds, team_bonus)

         # Save the draft results to a DataFrame
        draft_results = []
        for sim_num, draft in enumerate(all_drafts):
            for team, players in draft.items():
                result_entry = {
                    'Simulation': sim_num + 1,
                    'Team': team,
                }
                for i, player in enumerate(players):
                    result_entry.update({
                        f'G_{i+1}']': player['name'],
                        f'Player_{i+1}_Position': player['position'],
                        f'Player_{i+1}_Team': player['team']
                    })
                draft_results.append(result_entry)
        
        draft_results_df = pd.DataFrame(draft_results)
        
        # Display the results
        st.dataframe(draft_results_df)
        
        # Download link for the results
        csv = draft_results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Draft Results",
            data=csv,
            file_name='draft_results_with_team_stacking_and_positions.csv',
            mime='text/csv',
        )


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
            if i <= len(team_players):
                draft_results[idx, i - 1] = f"{team_players.iloc[i - 1]['G']}"
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

        # Rank teams
        ranks = total_points.argsort()[::-1].argsort() + 1

        # Assign payouts and accumulate them
        payouts = np.array([get_payout(rank) for rank in ranks])
        total_payouts += payouts

    # Calculate average payout per team
    avg_payouts = total_payouts / num_simulations
    return avg_payouts

def run_parallel_simulations(num_simulations, draft_results_df, projection_lookup):
    draft_results, teams = prepare_draft_results(draft_results_df)
    avg_payouts = simulate_team_projections(draft_results, projection_lookup, num_simulations)
    
    # Prepare final results
    final_results = pd.DataFrame({
        'Team': teams,
        'Average_Payout': avg_payouts
    })
    
    return final_results

def main():
    st.title("Fantasy Football Draft Simulator")

    # File uploader for projections
    projections_file = st.file_uploader("Choose a CSV file with player projections", type="csv")
    
    # File uploader for draft results
    draft_results_file = st.file_uploader("Choose a CSV file with draft results", type="csv")
    
    if projections_file is not None and draft_results_file is not None:
        projections_df = pd.read_csv(projections_file)
        draft_results_df = pd.read_csv(draft_results_file)
        
        st.write("Projections and draft results loaded successfully!")
        
        # Display a sample of the loaded data
        st.subheader("Sample of loaded projections:")
        st.write(projections_df.head())
        
        st.subheader("Sample of loaded draft results:")
        st.write(draft_results_df.head())

        # Create projection lookup dictionary
        projection_lookup = dict(zip(projections_df['name'], zip(projections_df['proj'], projections_df['projsd'])))

        # Input for number of simulations
        num_simulations = st.number_input("Number of simulations to run", min_value=100, max_value=100000, value=10000, step=100)

        # Button to start simulation
        if st.button("Run Simulations"):
            # Run simulations
            with st.spinner('Running simulations...'):
                final_results = run_parallel_simulations(num_simulations, draft_results_df, projection_lookup)

            # Display results
            st.subheader("Simulation Results:")
            st.write(final_results)

            # Option to download results
            csv = final_results.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="simulation_results.csv",
                mime="text/csv",
            )

if __name__ == '__main__':
    main()
