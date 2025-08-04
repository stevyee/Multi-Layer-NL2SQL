import os
import json
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
import streamlit as st

def generate_table_schema(output_dir="schemas"):
    """
    Generate a JSON schema categorizing database tables using AI by reading tableinfo.txt.
    
    Parameters:
    output_dir (str): Directory to save the schema files
    
    Returns:
    dict: The generated JSON schema
    """
    # Check if tableinfo.txt exists
    if not os.path.exists("tableinfo.txt"):
        st.error("tableinfo.txt not found. Please create this file with table descriptions.")
        return None
    
    # Read the table information from tableinfo.txt
    try:
        with open("tableinfo.txt", "r") as f:
            table_info = f.read()
    except Exception as e:
        st.error(f"Error reading tableinfo.txt: {e}")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"table_schema_{timestamp}.json")
    
    # Define a prompt template for grouping the tables
    prompt_template = """
You are given a list of database tables with their descriptions in the following format:
"table_name : description"

Your task is to group the tables by assigning each one a major label and a minor label for subgrouping.
Rules:
1. Each table must belong to one and only one major group and one minor subgroup.
2. Major groups should be high-level categories (for example, "User Management", "Bike Operations", "Geospatial Data", etc.).
3. Minor subgroups are more specific categories within each major group.
4. The output should be a valid JSON with the following structure:
{{
    "Major Group Label": {{
        "Minor Subgroup Label": ["table_name1", "table_name2", "..."],
        "Another Minor Subgroup": ["..."]
    }},
    "Another Major Group": {{
        "Some Minor Subgroup": ["..."]
    }}
}}
5. Make sure there is no duplication – each table appears only in one subgroup.
6. Your response must ONLY contain the JSON object and nothing else - no explanations, no markdown formatting.

Here is the list of tables:
{table_list}
"""

    # Create the prompt with the input variable
    prompt = PromptTemplate.from_template(prompt_template)
    
    # Initialize the LLM
    llm = ChatOpenAI(model="o1-mini", temperature=1)
    
    # Create the chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run the chain with the input
    result = chain.invoke({"table_list": table_info})
    
    # Extract the text from the result
    result_text = result["text"]
    
    # Try to clean and extract JSON from the text
    cleaned_text = result_text.strip()
    
    # Try to find JSON content within markdown code blocks
    if "```json" in cleaned_text and "```" in cleaned_text.split("```json", 1)[1]:
        cleaned_text = cleaned_text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned_text and "```" in cleaned_text.split("```", 1)[1]:
        cleaned_text = cleaned_text.split("```", 1)[1].split("```", 1)[0].strip()
    
    # Try to find content between curly braces if still not valid JSON
    if cleaned_text and not cleaned_text.startswith("{"):
        if "{" in cleaned_text and "}" in cleaned_text:
            start_idx = cleaned_text.find("{")
            end_idx = cleaned_text.rfind("}")
            if start_idx < end_idx:
                cleaned_text = cleaned_text[start_idx:end_idx+1]
    
    # Try to parse the cleaned output as JSON
    try:
        groups = json.loads(cleaned_text)
        
        # Save the JSON to a file with version control
        with open(output_file, "w") as f:
            json.dump(groups, f, indent=4)
        
        # Also save to a "latest" version for easy access
        latest_file = os.path.join(output_dir, "table_schema_latest.json")
        with open(latest_file, "w") as f:
            json.dump(groups, f, indent=4)
            
        print(f"Schema saved to {output_file}")
        print(f"Latest schema also available at {latest_file}")
        
        return groups
    except Exception as e:
        print("Error parsing JSON output:", e)
        st.error(f"Error parsing AI response as JSON: {e}")
        
        # Save the raw output to a text file for inspection
        debug_file = os.path.join(output_dir, f"raw_output_{timestamp}.txt")
        with open(debug_file, "w") as f:
            f.write(result_text)
        print(f"Raw output saved to {debug_file} for debugging")
        
        return None

def get_or_generate_table_schema(force_regenerate=False):
    """
    Get the existing table schema or generate it if it doesn't exist,
    using the tableinfo.txt file for table descriptions.
    
    Parameters:
    force_regenerate (bool): Whether to force regeneration of schema
    
    Returns:
    dict: The JSON schema of database tables
    """
    # Path to the latest schema file
    schema_dir = "schemas"
    latest_schema_path = os.path.join(schema_dir, "table_schema_latest.json")
    
    # Check if we already have a schema file and we're not forcing regeneration
    if os.path.exists(latest_schema_path) and not force_regenerate:
        try:
            with open(latest_schema_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading schema: {e}")
            st.error(f"Error loading existing schema: {e}")
            # Fall through to regeneration
    
    # Check if tableinfo.txt exists before trying to generate
    if not os.path.exists("tableinfo.txt"):
        st.error("tableinfo.txt not found. Please create this file with table descriptions.")
        return None
        
    # Generate new schema using tableinfo.txt
    return generate_table_schema(schema_dir)