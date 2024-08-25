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
