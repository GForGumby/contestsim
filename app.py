import streamlit as st
import pandas as pd
import numpy as np
from numba import jit

@jit(nopython=True)
def generate_projection(median, std_dev):
    fluctuation = np.random.uniform(-0.01, 0.01) * median
    return max(0, np.random.normal(median, std_dev) + fluctuation)

@jit(nopython=True)
def get_payout(rank):
    if rank == 1:
        return 100000.00
    elif rank == 2:
        return 35000.00
    # ... (rest of your payout structure)
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
