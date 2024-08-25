import streamlit as st
import pandas as pd
import numpy as np
from numba import jit, prange
from numba.np.random import randn, rand

@jit(nopython=True, parallel=True)
def generate_projections(medians, std_devs, num_simulations):
    result = np.empty((num_simulations, len(medians)))
    for i in prange(num_simulations):
        fluctuations = rand(len(medians)) * 0.02 - 0.01  # Uniform between -0.01 and 0.01
        projections = medians + randn(len(medians)) * std_devs + fluctuations * medians
        result[i] = np.maximum(0, projections)
    return result

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

@jit(nopython=True, parallel=True)
def simulate_team_projections(draft_results, projections, proj_indices, num_simulations):
    num_teams = draft_results.shape[0]
    total_payouts = np.zeros(num_teams)

    for sim in prange(num_simulations):
        total_points = np.zeros(num_teams)
        for i in range(num_teams):
            for j in range(6):
                player_index = proj_indices[draft_results[i, j]]
                if player_index != -1:
                    total_points[i] += projections[sim, player_index]

        ranks = np.argsort(np.argsort(-total_points)) + 1
        payouts = np.array([get_payout(rank) for rank in ranks])
        total_payouts += payouts

    return total_payouts / num_simulations

def run_parallel_simulations(num_simulations, draft_results_df, projection_lookup):
    teams = draft_results_df['Team'].unique()
    num_teams = len(teams)
    draft_results = np.empty((num_teams, 6), dtype='U50')

    for idx, team in enumerate(teams):
        team_players = draft_results_df[draft_results_df['Team'] == team]
        for i in range(6):
            if i < len(team_players):
                draft_results[idx, i] = team_players.iloc[i]['G']
            else:
                draft_results[idx, i] = "N/A"

    player_names = list(projection_lookup.keys())
    proj_indices = {name: i for i, name in enumerate(player_names)}
    proj_indices = {name: proj_indices.get(name, -1) for name in np.unique(draft_results)}
    
    medians, std_devs = zip(*[projection_lookup.get(name, (0, 0)) for name in player_names])
    projections = generate_projections(np.array(medians), np.array(std_devs), num_simulations)

    avg_payouts = simulate_team_projections(draft_results, projections, proj_indices, num_simulations)
    
    return pd.DataFrame({'Team': teams, 'Average_Payout': avg_payouts})

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

        # Check and select correct column names
        proj_columns = projections_df.columns.tolist()
        st.subheader("Select columns for projections:")
        name_col = st.selectbox("Select column for player names", proj_columns)
        proj_col = st.selectbox("Select column for projections", proj_columns)
        projsd_col = st.selectbox("Select column for projection standard deviations", proj_columns)

        # Create projection lookup dictionary
        try:
            projection_lookup = dict(zip(projections_df[name_col], 
                                         zip(projections_df[proj_col].astype(float), 
                                             projections_df[projsd_col].astype(float))))
        except ValueError as e:
            st.error(f"Error creating projection lookup: {e}. Please ensure the selected columns contain valid numeric data.")
            return

        # Input for number of simulations
        num_simulations = st.number_input("Number of simulations to run", min_value=100, max_value=100000, value=10000, step=100)

        # Button to start simulation
        if st.button("Run Simulations"):
            # Run simulations
            with st.spinner('Running simulations...'):
                try:
                    final_results = run_parallel_simulations(num_simulations, draft_results_df, projection_lookup)
                except Exception as e:
                    st.error(f"An error occurred during simulation: {e}")
                    return

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

