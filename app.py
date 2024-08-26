import streamlit as st
import pandas as pd
import numpy as np
from numba import jit, prange
import stripe

# Set your Stripe API key
stripe.api_key = st.secrets["stripe_api_key_test"] if st.secrets["testing_mode"] else st.secrets["stripe_api_key"]

def create_checkout_session():
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': st.secrets["stripe_price_id"],
                'quantity': 1,
            }],
            mode='payment',
            success_url=st.secrets["redirect_url_test"] if st.secrets["testing_mode"] else st.secrets["redirect_url"],
            cancel_url=st.secrets["redirect_url_test"] if st.secrets["testing_mode"] else st.secrets["redirect_url"],
        )
        return checkout_session
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Your existing functions (generate_projection, get_payout, prepare_draft_results, simulate_team_projections, run_parallel_simulations) remain unchanged

def main():
    st.title("Fantasy Football Draft Simulator")

    # Check if user has paid
    if not st.session_state.get("paid", False):
        st.write("Please pay to access the Fantasy Football Draft Simulator.")
        if st.button("Pay Now"):
            session = create_checkout_session()
            if session:
                st.write(f"[Proceed to payment](https://checkout.stripe.com/pay/{session.id})")
        return  # Exit the function if payment is not made

    # Rest of your main function remains unchanged
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
