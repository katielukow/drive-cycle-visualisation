import streamlit as st
import os

# Define the directory for pages
PAGES_DIR = 'pages'

st.cache_data.clear()

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

st.sidebar.title('Navigation')
selection = st.sidebar.radio('Go to', list(pages.keys()))

# Load and display the selected page
page = load_module(selection, pages[selection])
if hasattr(page, 'app'):
    page.app()
else:
    st.error(f'The page {selection} does not have an `app` function.')
