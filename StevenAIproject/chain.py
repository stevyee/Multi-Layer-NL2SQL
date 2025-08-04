import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, List
import numpy as np
from langchain.agents.agent_types import AgentType
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from sqlconnect import db_connection  # Updated import
from dotenv import load_dotenv
from table_selection_handler import get_current_df
from sqlalchemy import text
from langchain_openai import ChatOpenAI
import getpass
import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from openai import OpenAI
# 載入 .env 檔案中的環境變數
load_dotenv()


llmo1 = ChatOpenAI(
    model_name="o1-mini",
    temperature=1
)


def get_table_explanation(table_name, engine=None):
    """Get explanation for a table"""
    if engine is None:
        engine = db_connection.get_engine()  # Use the connection class
    
    if engine is None:
        return "Database connection not available"
        
    try:
        with engine.connect() as conn:
            # Get table description
            query = text(f"DESCRIBE `{table_name}`")
            result = conn.execute(query)
            columns = result.fetchall()
            
            # Format the column information
            column_info = "\n".join([f"- {col[0]}: {col[1]}" for col in columns])
            
            return f"Table {table_name} structure:\n{column_info}"
    except Exception as e:
        return f"Error getting table explanation: {str(e)}"

def get_top5_values(engine, table_name, column_name):
    """Get top 5 distinct values from a column"""
    if engine is None:
        return [], False
        
    try:
        with engine.connect() as conn:
            query = f"SELECT DISTINCT `{column_name}` FROM `{table_name}` LIMIT 6"
            result = conn.execute(text(query))
            values = [str(row[0]) for row in result.fetchall()]
            has_more = len(values) > 5
            return values[:5], has_more
    except Exception as e:
        print(f"Error getting values: {e}")
        return [], False


# 載入 .env 檔案中的環境變數
load_dotenv()


# response = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant"},
#             {"role": "user", "content": "Hello"},

# # "----------------------------------------------------------------------------------------------------------------------------------------"
# def router1(input): #modify the router
#     routermessage = [message.content for message in input]
#     if "no input" in routermessage:
#         return "END"
#     elif "distance_filter" in workers:
#         print("routing to distance_filter")
#         return "distance_filter"
#     elif "date_filter" in workers:
#         print("routing to date_filter")
#         return "date_filter"
#     elif "state_filter" in workers:
#         print("routing to state_filter")
#         return "state_filter"
#     if "END" in routermessage:
#         return "END"
    

# from langgraph.graph import END, MessageGraph
# graph = MessageGraph()
# #dfprccc = car_df
# testlist.clear()
# workers.clear()

# graph.add_node("filter1", filter1)
# graph.add_node("date_filter", date_filter)
# graph.add_node("state_filter", state_filter)
# graph.add_node("distance_filter", distance_filter)
# # graph.add_node("Customized_filter", Customized_filter)

 
# graph.add_conditional_edges(
#      "filter1",
#      router1,
#      {
#          "date_filter":"date_filter",
#          "state_filter":"state_filter",
#          "distance_filter":"distance_filter",
#         #  "Customized_filter":"Customized_filter",
#          "END": END
#      }
#  )      
# # graph.add_conditional_edges(
# #      "Customized_filter",
# #      Customized_filter,
# #      {
# #          "date_filter":"date_filter",
# #      }
# #  ) 

# graph.add_edge("date_filter", "filter1")   
# graph.add_edge("state_filter", "filter1")         
# graph.add_edge("distance_filter", "filter1")  
# # graph.add_edge("Customized_filter", "filter1")  


# graph.set_entry_point("filter1")

# testprogram = graph.compile()


# from IPython.display import Image, display

# display(Image(testprogram.get_graph().draw_mermaid_png()))