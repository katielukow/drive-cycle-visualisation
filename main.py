import streamlit as st
from data_loader import load_and_process_data
from pages import All_Data_app, Clustering_app, User_Behaviour_app

def main():
    # Load and process data once
    data = load_and_process_data()
    
    # Create a sidebar for navigation
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", ['All Data', 'Clustering', 'User Behaviour'])
    
    if selection == 'All Data':
        All_Data_app(data)
    elif selection == 'Page 2':
        Clustering_app(data)
    elif selection == 'Page 3':
        User_Behaviour_app(data)

if __name__ == "__main__":
    main()