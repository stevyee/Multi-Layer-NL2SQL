import streamlit as st
import json
import os
from pathlib import Path

def load_json_data(file_path=None, uploaded_file=None):
    """
    Load JSON data either from a file path or an uploaded file
    
    Parameters:
    file_path (str): Path to a local JSON file
    uploaded_file (UploadedFile): Streamlit uploaded file object
    
    Returns:
    dict: Parsed JSON data or None if loading fails
    """
    try:
        if uploaded_file is not None:
            # Handle uploaded file
            return json.loads(uploaded_file.getvalue().decode("utf-8"))
        elif file_path and os.path.exists(file_path):
            # Handle local file path
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            return None
    except json.JSONDecodeError:
        st.error("Invalid JSON file. Please check the file format.")
        return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def display_json_hierarchy(data):
    """
    Display the hierarchical JSON data in an expandable UI
    
    Parameters:
    data (dict): JSON data to display
    """
    if not data:
        st.warning("No JSON data to display")
        return
    
    for category, subcategories in data.items():
        with st.expander(f"🔍 {category}", expanded=False):
            if isinstance(subcategories, dict):
                for subcategory, items in subcategories.items():
                    st.markdown(f"**📁 {subcategory}**")
                    if isinstance(items, list):
                        for item in items:
                            st.write(f"📄 {item}")
                    else:
                        st.write(f"📄 {items}")
            elif isinstance(subcategories, list):
                for item in subcategories:
                    st.write(f"📄 {item}")
            else:
                st.write(f"📄 {subcategories}")

def display_json_hierarchy_with_selection(data, table_info=None, handle_table_selection=None, toggle_table=None):
    """
    Display the hierarchical JSON data in an expandable UI with table selection capability
    
    Parameters:
    data (dict): JSON data to display
    table_info (dict): Database table information
    handle_table_selection (function): Function to handle table selection
    toggle_table (function): Function to toggle table expansion
    """
    if not data:
        st.warning("No JSON data to display")
        return
    
    # Used to track tables we've already seen to prevent duplicate keys
    seen_tables = set()
    
    for category_idx, (category, subcategories) in enumerate(data.items()):
        with st.expander(f"🔍 {category}", expanded=False):
            if isinstance(subcategories, dict):
                for subcategory_idx, (subcategory, items) in enumerate(subcategories.items()):
                    st.markdown(f"**📁 {subcategory}**")
                    if isinstance(items, list):
                        # Create columns for each item in the list
                        for item_idx, item in enumerate(items):
                            # Check if this item is a database table
                            if table_info and item in table_info:
                                info = table_info[item]
                                
                                # Create a unique key for this table based on its position in the JSON
                                unique_key = f"table_{category_idx}_{subcategory_idx}_{item_idx}_{item}"
                                
                                # Create a row with checkbox and table name button
                                col1, col2 = st.columns([0.15, 0.85])
                                with col1:
                                    # Use the standard checkbox key for state tracking
                                    checkbox_key = f"selected_table_{item}"
                                    
                                    # Track if we've seen this table before
                                    if item in seen_tables:
                                        # If we've seen this table before, use a unique key for the UI element
                                        # but maintain the same underlying session state
                                        st.checkbox(
                                            label=f"Select table {item}",
                                            key=unique_key,
                                            value=st.session_state.get(checkbox_key, False),
                                            on_change=handle_table_selection,
                                            args=(item,),
                                            label_visibility="collapsed"
                                        )
                                    else:
                                        # First time seeing this table, use the standard key
                                        if checkbox_key not in st.session_state:
                                            st.session_state[checkbox_key] = False
                                        
                                        st.checkbox(
                                            label=f"Select table {item}",
                                            key=checkbox_key,
                                            on_change=handle_table_selection,
                                            args=(item,),
                                            label_visibility="collapsed",
                                            value=st.session_state[checkbox_key]
                                        )
                                        seen_tables.add(item)
                                
                                with col2:
                                    if st.button(f"📄 {item} ({info.get('column_count', 0)}cols, {info.get('row_count', 0):,}rows)",
                                               key=f"btn_{unique_key}"):
                                        toggle_table(item)
                            else:
                                # Regular JSON item (not a table)
                                st.write(f"📄 {item}")
                    else:
                        st.write(f"📄 {items}")
            elif isinstance(subcategories, list):
                for item_idx, item in enumerate(subcategories):
                    # Check if this item is a database table
                    if table_info and item in table_info:
                        info = table_info[item]
                        
                        # Create a unique key for this table based on its position in the JSON
                        unique_key = f"table_{category_idx}_direct_{item_idx}_{item}"
                        
                        # Create a row with checkbox and table name button
                        col1, col2 = st.columns([0.15, 0.85])
                        with col1:
                            # Use the standard checkbox key for state tracking
                            checkbox_key = f"selected_table_{item}"
                            
                            # Track if we've seen this table before
                            if item in seen_tables:
                                # If we've seen this table before, use a unique key for the UI element
                                # but maintain the same underlying session state
                                st.checkbox(
                                    label=f"Select table {item}",
                                    key=unique_key,
                                    value=st.session_state.get(checkbox_key, False),
                                    on_change=handle_table_selection,
                                    args=(item,),
                                    label_visibility="collapsed"
                                )
                            else:
                                # First time seeing this table, use the standard key
                                if checkbox_key not in st.session_state:
                                    st.session_state[checkbox_key] = False
                                
                                st.checkbox(
                                    label=f"Select table {item}",
                                    key=checkbox_key,
                                    on_change=handle_table_selection,
                                    args=(item,),
                                    label_visibility="collapsed",
                                    value=st.session_state[checkbox_key]
                                )
                                seen_tables.add(item)
                        
                        with col2:
                            if st.button(f"📄 {item} ({info.get('column_count', 0)}cols, {info.get('row_count', 0):,}rows)", 
                                       key=f"btn_{unique_key}"):
                                toggle_table(item)
                    else:
                        # Regular JSON item (not a table)
                        st.write(f"📄 {item}")
            else:
                st.write(f"📄 {subcategories}")

def search_json_data(data, query):
    """
    Search JSON data for a specific query string
    
    Parameters:
    data (dict): JSON data to search through
    query (str): Search term
    
    Returns:
    dict: Subset of the JSON data containing matches
    """
    if not query or not data:
        return {}
        
    search_results = {}
    query = query.lower()
    
    for category, subcategories in data.items():
        if query in category.lower():
            search_results[category] = subcategories
            continue
            
        if isinstance(subcategories, dict):
            category_matches = {}
            for subcategory, items in subcategories.items():
                if query in subcategory.lower():
                    category_matches[subcategory] = items
                    continue
                    
                if isinstance(items, list):
                    item_matches = [item for item in items if isinstance(item, str) and query in item.lower()]
                    if item_matches:
                        category_matches[subcategory] = item_matches
                        
            if category_matches:
                search_results[category] = category_matches
    
    return search_results