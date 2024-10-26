import streamlit as st
import pandas as pd
import numpy as np
from numba import jit, prange
from st_paywall import add_auth
import io
from openpyxl import load_workbook

st.title("GUMBY SIMS- The first publicly available contest sim tool on Underdog. Take your game to the next level by simming out the highest ROI lineups,  optimal exposure percentages, and chalkiest field combos. ")

st.write("TO MANAGE YOUR SUBSCRIPTION GO HERE https://billing.stripe.com/p/login/9AQ4jFeRDbHTaIMfYY")
st.write("Please watch the entirety of this video explaining the product before purchasing (https://www.youtube.com/watch?v=3jrnl_yYkXs). I am available on twitter @GumbyUD for any questions or concerns about the product.")
st.write("On a Dawg Bowl sized slate, it takes about 2 mins to run the draft sim, and about 5 mins to run 5000 instances of the projection sim.")
st.write("For the team stacking bonus, use .99 if you want stack frequency to mimic real drafts. Use .98 if you want slightly more, and 1.00 for no stacking. The lower it is the more frequent stacks are in your field of lineups. I wouldn't make it lower than .95 except for MLB, which I am currently still testing.") 
st.write("Current supported sports: NFL main BR, Dawg Bowl.")


add_auth(required=True)

# Define the name of your Excel file
excel_file_name = 'Blank Analysis Template.xlsx'

# Define the name of your Excel file
excel_file_name = 'Blank Analysis Template.xlsx'

# Attempt to read the Excel file
try:
    with open(excel_file_name, "rb") as file:
        excel_data = file.read()
    
    # Create a download button
    st.download_button(
        label="Download Blank Analysis Template",
        data=excel_data,
        file_name=excel_file_name,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    st.success(f"'{excel_file_name}' is ready for download!")
except FileNotFoundError:
    st.error(f"Error: The file '{excel_file_name}' was not found.")
    st.info("Please make sure the file is in the same directory as your Streamlit app.")
except Exception as e:
    st.error(f"An error occurred while reading the file: {str(e)}")





st.write("Paste your sim results and draft results into the above file for more automated analysis")




st.subheader("NFL BR WEEK 8")

st.write("If you prefer to use my already completed drafts, download the draft results and analysis here (updated 10/3)!")

# Define the name of your Excel file
excel_file_name = 'Week 7 Analysis.xlsx'

# Attempt to read the Excel file
try:
    with open(excel_file_name, "rb") as file:
        excel_data = file.read()
    
    # Create a download button
    st.download_button(
        label="Download Week 8 Analysis 10/26 Update",
        data=excel_data,
        file_name=excel_file_name,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    st.success(f"'{excel_file_name}' is ready for download!")
except FileNotFoundError:
    st.error(f"Error: The file '{excel_file_name}' was not found.")
    st.info("Please make sure the file is in the same directory as your Streamlit app.")
except Exception as e:
    st.error(f"An error occurred while reading the file: {str(e)}")



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
        return 30000.00
    elif rank == 2:
        return 15000.00
    elif rank == 3:
        return 7500.00
    elif rank == 4:
        return 6000.00
    elif rank == 5:
        return 5000.00
    elif rank == 6:
        return 4250.00
    elif rank == 7:
        return 3750.00
    elif rank == 8:
        return 3500.00
    elif rank == 9:
        return 3250.00
    elif rank == 10:
        return 3000.00
    elif 11 <= rank <= 25:
        return 1000.00
    elif 26 <= rank <= 50:
        return 500.00
    elif 51 <= rank <= 100:
        return 125.00
    elif 101 <= rank <= 200:
        return 55.00
    elif 201 <= rank <= 500:
        return 35.00
    elif 501 <= rank <= 1000:
        return 25.00
    elif 1001 <= rank <= 2000:
        return 20.00
    elif 2001 <= rank <= 3000:
        return 15.00
    elif 3001 <= rank <= 7500:
        return 12.00
    elif 7501 <= rank <= 14250:
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
            if player_col in team_players.columns:
                draft_results[idx, i - 1] = team_players[player_col].iloc[0]
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
st.title("NFL Fantasy Football Custom Draft Simulator")

sample_csv_path = 'NFL WEEK 4 ADP TEMPLATE.csv'
with open(sample_csv_path, 'rb') as file:
    sample_csv = file.read()

st.download_button(
    label="Download NFL WEEK 4 ADP TEMPLATE",
    data=sample_csv,
    file_name='NFL WEEK 4 ADP TEMPLATE.csv',
    mime='text/csv',
    key="nfl_week2_adp_template_download"
)

# File upload for ADP
adp_file = st.file_uploader("Upload your NFL ADP CSV file", type=["csv"], key="adp_nfl_uploader")

if adp_file is not None:
    df = pd.read_csv(adp_file)
    if 'player_id' not in df.columns:
        df['player_id'] = df.index
    
    num_simulations = st.number_input("Number of simulations", min_value=1, value=10, key="nflnum_simulations_input")
    num_teams = st.number_input("Number of teams", min_value=2, value=6, key="nflnum_teams_input")
    num_rounds = st.number_input("Number of rounds", min_value=1, value=6, key="nflnum_rounds_input")
    team_bonus = st.number_input("Team stacking bonus", min_value=0.0, value=0.99, key="nflteam_bonus_input")
    
    if st.button("Run Draft Simulation", key="nflrun_draft_sim_button"):
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
        
        csv = draft_results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download NFL Draft Results",
            data=csv,
            file_name='nfl_draft_results.csv',
            mime='text/csv',
            key="nfldownload_draft_results_button"
        )

# File uploaders for projections and draft results
projections_file = st.file_uploader("Choose a CSV with NFL player projections", type="csv", key="nflprojections_uploader")
draft_results_file = st.file_uploader("Choose a CSV file with NFL draft results", type="csv", key="nfldraft_results_uploader")

if projections_file is not None and draft_results_file is not None:
    projections_df = pd.read_csv(projections_file)
    draft_results_df = pd.read_csv(draft_results_file)

    projection_lookup = dict(zip(projections_df['name'], zip(projections_df['proj'], projections_df['projsd'])))

    num_simulations = st.number_input("Number of simulations to run", min_value=100, max_value=100000, value=10000, step=100, key="nflnum_proj_simulations_input")

    if st.button("Run Projection Simulations", key="nflrun_proj_sim_button"):
        with st.spinner('Running simulations...'):
            final_results = run_parallel_simulations(num_simulations, draft_results_df, projection_lookup)

        csv = final_results.to_csv(index=False)
        st.download_button(
            label="Download Projection Results as CSV",
            data=csv,
            file_name="projection_simulation_results.csv",
            mime="text/csv",
            key="nfldownload_proj_results_button"
        )
st.subheader("------------------------------------------------------------------------------")

st.subheader("NFL Post Contest Sim Data")
st.write("Week 6")

# Define the name of your Excel file
excel_file_name = 'Week 6 Post Contest Sim.xlsx'

# Attempt to read the Excel file
try:
    with open(excel_file_name, "rb") as file:
        excel_data = file.read()
    
    # Create a download button
    st.download_button(
        label="Download Week 6 Post Contest Sim",
        data=excel_data,
        file_name=excel_file_name,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    st.success(f"'{excel_file_name}' is ready for download!")
except FileNotFoundError:
    st.error(f"Error: The file '{excel_file_name}' was not found.")
    st.info("Please make sure the file is in the same directory as your Streamlit app.")
except Exception as e:
    st.error(f"An error occurred while reading the file: {str(e)}")




st.write("Week 5")

# Define the name of your Excel file
excel_file_name = 'Week 5 Post Contest Sim.xlsx'

# Attempt to read the Excel file
try:
    with open(excel_file_name, "rb") as file:
        excel_data = file.read()
    
    # Create a download button
    st.download_button(
        label="Download Week 5 Post Contest Sim",
        data=excel_data,
        file_name=excel_file_name,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    st.success(f"'{excel_file_name}' is ready for download!")
except FileNotFoundError:
    st.error(f"Error: The file '{excel_file_name}' was not found.")
    st.info("Please make sure the file is in the same directory as your Streamlit app.")
except Exception as e:
    st.error(f"An error occurred while reading the file: {str(e)}")

st.write("Week 4")

# Define the name of your Excel file
excel_file_name = 'Week 4 Post Contest Sim.xlsx'

# Attempt to read the Excel file
try:
    with open(excel_file_name, "rb") as file:
        excel_data = file.read()
    
    # Create a download button
    st.download_button(
        label="Download Week 4 Post Contest Sim",
        data=excel_data,
        file_name=excel_file_name,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    st.success(f"'{excel_file_name}' is ready for download!")
except FileNotFoundError:
    st.error(f"Error: The file '{excel_file_name}' was not found.")
    st.info("Please make sure the file is in the same directory as your Streamlit app.")
except Exception as e:
    st.error(f"An error occurred while reading the file: {str(e)}")

st.write("Week 3")

# Define the name of your Excel file
excel_file_name = 'Week 3 Post Contest Sim.xlsx'

# Attempt to read the Excel file
try:
    with open(excel_file_name, "rb") as file:
        excel_data = file.read()
    
    # Create a download button
    st.download_button(
        label="Download Week 3 Post Contest Sim",
        data=excel_data,
        file_name=excel_file_name,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    st.success(f"'{excel_file_name}' is ready for download!")
except FileNotFoundError:
    st.error(f"Error: The file '{excel_file_name}' was not found.")
    st.info("Please make sure the file is in the same directory as your Streamlit app.")
except Exception as e:
    st.error(f"An error occurred while reading the file: {str(e)}")

