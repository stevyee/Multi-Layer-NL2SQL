from sqlconnect import db_connection
from sqlalchemy import text
import pandas as pd

# Global variable to store the DataFrame
current_df = None
tablename = None
def confirm_selected_tables(selected_tables):
    """
    接收用户选择的表（一个 set 对象），并返回一个列表。
    后续可以在这里增加额外的处理逻辑（例如验证、日志记录等）。
    """
    global current_df
    global table_name
    print("Selected tables:", selected_tables)
    table_name = str(selected_tables)
    confirmed_tables = []
    
    # Convert set to list if it's not already
    tables_to_process = list(selected_tables)
    
    # Create an empty DataFrame to store all tables
    all_dfs = []
    
    for table_name in tables_to_process:
        try:
            df = pd.read_sql_table(table_name, db_connection.get_engine())
            print(f"Preview of {table_name}:")
            print(df.head(3))
            all_dfs.append(df)
            confirmed_tables.append(table_name)
        except Exception as e:
            print(f"Error processing table {table_name}: {e}")
    
    # Combine all DataFrames if there are any
    
    current_df = pd.concat(all_dfs, axis=1)
    
    
    return confirmed_tables

def get_current_df():
    """
    Returns the current DataFrame for use in other modules
    """
    global current_df
    return current_df
