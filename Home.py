import streamlit as st
import os

# Define the directory for pages
PAGES_DIR = 'pages'

# st.cache_data.clear()
# file_path = os.path.join(PAGES_DIR, '1_All_Data.py')

# Function to load a module from a file
def load_module(module_name, filepath):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Get a list of pages
page_files = [f for f in os.listdir(PAGES_DIR) if f.endswith('.py')]
pages = {f.replace('.py', '').replace('_', ' ').title(): os.path.join(PAGES_DIR, f) for f in page_files}

main_page = '0 Trip Visualisation'
load_module(main_page, pages[main_page]).app()
