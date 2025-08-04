from langchain_openai import ChatOpenAI
import streamlit as st
import pandas as pd
import os
import re
import shared
import tempfile
import subprocess
from table_selection_handler import get_current_df
import json_handler
from sqlalchemy import text, Table, MetaData  
from chain import get_table_explanation
from table_selection_handler import confirm_selected_tables
from sqlconnect import db_connection, get_engine
from location_matcher import LocationMatcher
from datetime import datetime


# Initialize location matcher at the beginning of your application
location_matcher = None

def init_location_matcher():
    """Initialize the location matcher with knowledge base"""
    global location_matcher
    
    # Path to your location knowledge base
    kb_path = 'locations.json'
    
    # If the knowledge base doesn't exist, create it (optional)
    
    # Initialize the matcher
    location_matcher = LocationMatcher(kb_path)
    return location_matcher
llmo1 = ChatOpenAI(model="o1-mini",temperature=1)
testlist = []

# -------------------------------
# 新增：用于获取指定表中仅包含 datetime 类型字段的函数
# 解释：此函数会连接数据库，对指定表执行 DESCRIBE，
# 并将所有类型为 datetime 的字段名存入全局变量 tablename，
# 该变量随后可供 agentest 模块使用
# -------------------------------
def get_table_description_only_datetime(engine, table_name):
    """
    返回指定表中所有 datetime 类型的字段名称列表
    """
    describe = {}
    datetime_fields = []
    with engine.connect() as conn:
        result = conn.execute(text("DESCRIBE " + str(table_name)))
        for row in result:
            # 将每一行的第一个元素作为键，第二个元素作为值存入字典
            describe[row[0]] = row[1]
    for column, col_type in describe.items():
        if col_type.lower() == "datetime":
            datetime_fields.append(column)
        elif col_type.lower() == "timestamp":
            datetime_fields.append(column)
    global tablename 
    tablename = datetime_fields  # 更新全局变量 tablename
    return tablename

def get_table_description_only_region(engine, table_name):
    """
    返回指定表中所有涉及location的字段名称列表
    """
    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=engine)
    # 获取所有列名
    columns = table.columns.keys()
    
    # # 执行查询获取一行示例数据
    # stmt = select(table).limit(1)
    # with engine.connect() as conn:
    #     sample_data = conn.execute(stmt).first()
    
    prompt = """
this is all the columns in my table: """+str(columns)+"""
and now, you need to identify all the columns that are possiblily related to describing locations,
ONLY return the column names and nothing else.
strictly follow the below output format:
keyword1, keyword2, ...etc
"""

    res = llmo1.invoke(prompt)
    result_region = res.content.strip().split(", ")
    # 执行查询获取一行示例数据
    global tablename 
    tablename = result_region 
    print("this is the region keyword from the function:"+str(tablename))# 更新全局变量 tablename
    return tablename

def show_keyword_selection_interface():
    if st.session_state.selected_tables and st.session_state.get("engine") is not None:
        selected_table = list(st.session_state.selected_tables)[0]  # 取第一个选定表
        
        # 获取日期和区域字段
        if "dt_fields" not in st.session_state or "region_fields" not in st.session_state:
            # 首次加载时获取字段信息
            st.session_state.dt_fields = get_table_description_only_datetime(st.session_state["engine"], selected_table)
            st.session_state.region_fields = get_table_description_only_region(st.session_state["engine"], selected_table)
            
        # 初始化session state变量
        if "selected_dt_field" not in st.session_state:
            st.session_state.selected_dt_field = None
        if "selected_region_fields" not in st.session_state:
            st.session_state.selected_region_fields = []
        
        # 显示日期字段选择 - 使用横向布局
        st.write("### Please select date/time keywords(max 1):")
        
        # 计算每行显示的按钮数量（最多4个按钮一行）
        dt_fields = st.session_state.dt_fields if st.session_state.dt_fields else []
        dt_buttons_per_row = min(4, len(dt_fields))
        
        # 如果有日期字段，创建按钮
        if dt_fields:
            # 分割成行来显示按钮
            for i in range(0, len(dt_fields), dt_buttons_per_row):
                # 获取这一行的字段
                row_fields = dt_fields[i:i + dt_buttons_per_row]
                cols = st.columns(len(row_fields))
                
                for j, field in enumerate(row_fields):
                    # 如果当前已选择该字段，则给按钮加上标记
                    button_text = f"✓ {field}" if st.session_state.selected_dt_field == field else field
                    
                    if cols[j].button(button_text, key=f"dt_{field}"):
                        # 只能选择一个日期字段，切换选择状态
                        if st.session_state.selected_dt_field == field:
                            st.session_state.selected_dt_field = None
                        else:
                            st.session_state.selected_dt_field = field
                        st.rerun()
        else:
            st.write("No time/date keyword avaliable")
        
        # 显示当前选择的日期字段
        st.write(f"Chosen time/date keywords: {st.session_state.selected_dt_field if st.session_state.selected_dt_field else '未选择'}")
        
        # 显示区域字段选择 - 使用横向布局
        st.write("### Please select regional/location keywords(max 2):")
        
        # 计算每行显示的按钮数量（最多4个按钮一行）
        region_fields = st.session_state.region_fields if st.session_state.region_fields else []
        region_buttons_per_row = min(4, len(region_fields))
        
        # 如果有区域字段，创建按钮
        if region_fields:
            # 分割成行来显示按钮
            for i in range(0, len(region_fields), region_buttons_per_row):
                # 获取这一行的字段
                row_fields = region_fields[i:i + region_buttons_per_row]
                cols = st.columns(len(row_fields))
                
                for j, field in enumerate(row_fields):
                    # 如果当前已选择该字段，则给按钮加上标记
                    is_selected = field in st.session_state.selected_region_fields
                    button_text = f"✓ {field}" if is_selected else field
                    
                    if cols[j].button(button_text, key=f"rg_{field}"):
                        # 切换选择状态
                        if field in st.session_state.selected_region_fields:
                            st.session_state.selected_region_fields.remove(field)
                        else:
                            if len(st.session_state.selected_region_fields) < 2:  # 限制最多选择两个区域字段
                                st.session_state.selected_region_fields.append(field)
                        st.rerun()
        else:
            st.write("No region/location keyword avaliable")
        
        # 显示当前选择的区域字段
        selected_regions = ", ".join(st.session_state.selected_region_fields) if st.session_state.selected_region_fields else "未选择"
        st.write(f"Chosen region/location keywords: {selected_regions}")
        
        return True
    else:
        st.error("Please select a table and ensure db connection")
        return False

# -------------------------------
# 以下是原有代码：session_state 初始化、UI 布局、数据库概览等
# -------------------------------

describe = {}
tablename = None
datetime_fields = []
if "confirmed_tables" not in st.session_state:
    st.session_state.confirmed_tables = []
if "keywords_confirmed" not in st.session_state:
    st.session_state.keywords_confirmed = False

# 初始化选择的表和列
if "selected_tables" not in st.session_state:
    st.session_state.selected_tables = set()
if "selected_columns" not in st.session_state:
    st.session_state.selected_columns = set()
if "messages" not in st.session_state:
    st.session_state.messages = []

# 初始化数据库概览
if "db_overview_expanded" not in st.session_state:
    st.session_state.db_overview_expanded = {}
if "engine" not in st.session_state:
    st.session_state.engine = None
if "db_overview" not in st.session_state:
    st.session_state.db_overview = {}

if "connection_settings" not in st.session_state:
    st.session_state.connection_settings = {
        "ssh_address": "52.74.219.215",
        "ssh_port": 22,
        "ssh_username": "ec2-user",
        "remote_bind_address": "loco-erp-dbinstance.cf8ccanlhjrl.ap-east-1.rds.amazonaws.com",
        "remote_bind_port": 3306,
        "db_user": "steven",
        "db_password": "p94oha6yFmr_nKDq",
        "db_name": "locoerp"
    }

def compile_and_run(code: str) -> str:
    
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as tmp:
        tmp.write(code)
        tmp.flush()
        temp_file_path = tmp.name

    try:
        result = subprocess.run(
            ['python', temp_file_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        # 如果 stdout 为空，返回 stderr 以便调试错误


        return result.stdout if result.stdout.strip() else result.stderr
    except Exception as e:
        return f"代码执行出错：{str(e)}"

def save_table_to_csv(df, filename):
    """Save DataFrame to CSV if it exists"""
    if df is not None and isinstance(df, pd.DataFrame):
        try:
            df.to_csv(filename, index=False)
            return True
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            return False
    return False
# -------------------------------
# 修改：在调用 AI 回答之前调用 get_table_description_only_datetime
# 解释：在 get_ai_response 函数中，如果已经选定了表且已连接数据库，
# 则提取该表的 datetime 字段信息，并输出以供调试或后续使用
# -------------------------------
def classify_query(user_input):
    """
    调用 AI 分类助手对用户输入进行分类，返回分类字符串
    """
    classification_message = (
        """你是一个智能分类助手，负责分析用户的提问并识别其中的关键信息。你的任务是从用户输入中提取以下三类关键词：
1. **地区关键词**：与地理位置相关的词汇（例如城市、区、街道等）。
2. **时间关键词**：与时间相关的词汇（例如日期、月份、时间段等）。
3. **通用查询**：不属于上述两类的其他问题或请求。

根据提取到的关键词类型，你需要判断需要将数据传递给以下哪些 Worker 进行处理：
- 如果包含地区关键词，如区域，区，站，或其他地区有关的询问关键词如附近，周围等，则需要传递给 **region_filter**。
- 如果包含时间关键词，如明天，昨天，以及具体的日期和时间或与其他时间有关的询问关键词，则需要传递给 **date_filter**。
- 否则，则需要传递给 **general_worker**。

请按照以下格式回答：
- 如果需要传递给多个 Worker，请列出所有相关的 Worker，例如：“region_filter,date_filter,general_worker”。
- 如果只涉及某一个 Worker，则仅列出该 Worker，例如：“general_worker”。

现在，请分析以下用户输入并给出你的回答：

用户输入：""" + str(user_input)
    )
    # classification_message = ("always return 'region_filter,date_filter,general_worker' no matter what, and strictly follow the format")
    classification_response = llmo1.invoke(classification_message).content
    st.write("Classification returns:", classification_response)
    return classification_response
def parse_classification(result_from_classify_query):
    """
    解析分类助手的返回字符串，返回是否涉及时间、区域以及通用查询的标志位
    """
    workers = [w.strip() for w in result_from_classify_query.split(",")]
    date_related = "date_filter" in workers
    region_related = "region_filter" in workers
    general_worker = "general_worker" in workers
    return date_related, region_related, general_worker

def generate_modified_description(user_input, selected_table, dt_field, region_fields, date_related, region_related):
    """
    Generate a modified description based on classification results and table information
    while preserving the original query intent
    """
    print("The datetime keywords are:" + str(dt_field))
    print("The region keywords are:" + str(region_fields))
    
    # Identify key query intents to preserve
    query_intent = ""
    intent_keywords = {
        "peak hour": "find the most frequent or common hour (peak hour)",
        "busiest time": "find the most frequent or common hour (peak hour)",
        "rush hour": "find the most frequent or common hour (peak hour)",
        "most popular": "find the most frequent or common values",
        "traffic flow": "analyze the traffic patterns or flow",
        "distance": "calculate and filter by distance"
    }
    
    # Check if any intent keywords are in the user input
    for keyword, description in intent_keywords.items():
        if keyword.lower() in user_input.lower():
            query_intent = f"The goal is to {description}. "
            break
    
    # Build the modified description with preserved intent
    if date_related and region_related:
        modified_description = (
            f"{user_input}, use '{dt_field}' as time/date keywords in table: '{selected_table}' "
            f"and also use '{region_fields}' as regional/location keywords in table: '{selected_table}'. "
            f"{query_intent}Make sure to properly analyze time patterns if the query is about peak hours or time trends."
        )
    elif date_related:
        modified_description = (
            f"{user_input}, use '{dt_field}' as time/date keywords in table: '{selected_table}'. "
            f"{query_intent}Make sure to properly analyze time patterns if the query is about peak hours or time trends."
        )
    elif region_related:
        modified_description = (
            f"{user_input}, use '{region_fields}' as regional/location keywords in table: '{selected_table}'. "
            f"{query_intent}If the query involves time patterns, extract the hour component for analysis."
        )
    else:
        # Modify this part to add customized extracted columns
        modified_description = (
            f"{user_input}, enquire the data in table: '{selected_table}'. "
            f"{query_intent}"
        )
    
    # For peak hour queries, add explicit instructions
    if "peak hour" in user_input.lower() or "busiest time" in user_input.lower():
        modified_description += (
            " Specifically, you need to: "
            "1) Convert timestamp fields to datetime format, "
            "2) Extract the hour component, "
            "3) Count occurrences by hour, "
            "4) Identify and report the hour with the highest count as the peak hour."
        )
        
    return modified_description

# Add initialization for keyword confirmation flag


def dateandregion_filter(user_input):
    """
    Process user queries with improved location handling that preserves the original query structure.
    """
    # Check if there are selected tables and the database connection is valid
    if st.session_state.selected_tables and st.session_state.get("engine") is not None:
        selected_table = list(st.session_state.selected_tables)[0]  # Get the first selected table
        
        # Get the keywords from session state
        selected_dt = st.session_state.get("selected_dt_field")
        selected_rg = st.session_state.get("selected_region_fields")
        
        # Initialize location matcher if not already done
        global location_matcher
        if location_matcher is None:
            location_matcher = init_location_matcher()
        
        # Step 1: Extract location keywords using AI
        from extract_location_agent import extract_location_keyword
        location_keywords = extract_location_keyword(user_input)
        
        # Create variables to store the location information
        location_name = None
        location_coords = None
        match_score = None
        
        # Step 2: Match extracted locations with knowledge base
        if location_keywords:
            # Find the best match for each extracted location
            best_match = None
            best_score = 0
            
            for keyword in location_keywords:
                # Simple match attempt first
                if keyword in location_matcher.locations:
                    location_name = keyword
                    location_coords = location_matcher.locations[keyword]
                    match_score = 100
                    break
                
                # Otherwise, try fuzzy matching
                match_result = location_matcher.find_best_location_match(keyword)
                if match_result and match_result[2] > best_score:
                    location_name, location_coords, match_score = match_result
                    best_score = match_score
            
            # Display matched location information
            if location_name:
                st.info(f"Location Keyword extracted: '{location_keywords}'")
                st.info(f"Matched location '{location_name}' to coordinates: {location_coords[0]}, {location_coords[1]} (Match score: {match_score}%)")
                
        # Step 3: Call the classification on the ORIGINAL user input (not modified)
        classify = classify_query(user_input)
        date_related, region_related, general_worker = parse_classification(classify)
        
        # Step 4: Generate the modified description using the selected keywords
        # Use the enhanced generate_modified_description function we created earlier
        description = generate_modified_description(
            user_input,  # Use original query
            selected_table, 
            selected_dt, 
            selected_rg,
            date_related,
            region_related
        )
        
        # Step 5: Add location coordinates to the description if we found them
        if location_coords:
            # Add target coordinates to the description without modifying the original query
            description += f" Use target coordinates: latitude={location_coords[0]}, longitude={location_coords[1]} for any distance calculations."
        
        # Display the keywords being used
        st.write(f"Using time field: {selected_dt}")
        st.write(f"Using region fields: {selected_rg}")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("query_log.txt", "a") as log_file:
            log_file.write(f"[{current_time}] Original: \"{user_input}\", Processed: \"{location_keywords}\"\n")
        
        
        # Save table to CSV
        table = get_current_df()  # Local DataFrame
        save_table_to_csv(table, "data.csv")
        
        # Initialize AI model
        llmo1 = ChatOpenAI(model='o1-mini', temperature=1)
        
        # Generate code based on the enhanced description
        code_generation_prompt = f"""
Task: Generate Python code to solve the user's question about pandas DataFrame based on the following description. Please ensure the code is runnable and output the results at the end.
Requirements:
1. Must use the entire DataFrame, no random sampling allowed.
2. Import data from the CSV file 'data.csv' using pandas and assign it to variable df.
3. Filter the data according to the problem description and output the filtered results.
4. If the problem involves latitude and longitude, use the target coordinates provided in the description.
5. If the query is about "peak hour" or "busiest time", make sure to:
   a) Convert timestamp fields to datetime format using pd.to_datetime
   b) Extract the hour component using dt.hour
   c) Calculate frequency of each hour
   d) Identify and report the hour with the highest count

Problem description: {description}
        """
        
        generated_response = llmo1.invoke(code_generation_prompt)
        response_content = generated_response.content
        
        # Try to extract code in markdown format
        code_match = re.search(r"```python(.*?)```", response_content, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            code = response_content.strip()
        
        # Create a temporary file to store the code
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as tmp:
            tmp.write(code)
            tmp.flush()  # Ensure data is written to the file
            temp_file_path = tmp.name
        
        print("Generated temporary file path:", temp_file_path)
        
        # Call the executor to run the generated code
        execution_result = compile_and_run(code)
        result = execution_result.splitlines()
        
        # Generate analysis based on the execution result
        analysis_prompt = f"""You are given this user query: {user_input} and this is the result of the agent executing the program generated from the user query: {result}, 
please consider both the user query and program result and give an insightful conclusion and implement analysis and suggestion for future decision making in English"""
        
        analysis = llmo1.invoke(analysis_prompt)   
        print(analysis.content)
        
        return analysis.content



# def main_router(user_input):
#     return


def get_ai_response(user_input):

    return dateandregion_filter(user_input)



# -----------------------------------------------------------------------------
# 2. 获取数据库概览（表->列）仅执行一次
# -----------------------------------------------------------------------------
def get_db_overview(engine):
    """
    返回数据库的整体信息：{
        "table_name": {
            "columns": ["col1", "col2", ...],
            "row_count": 100,
            "column_count": 5
        },
        ...
    }
    """
    overview = {}
    try:
        with engine.connect() as conn:
            tables = conn.execute(text("SHOW TABLES")).fetchall()
            print(f"Found tables: {[table[0] for table in tables]}")
            for table_row in tables:
                table_name = table_row[0]
                try:
                    columns_info = conn.execute(text(f"DESCRIBE `{table_name}`")).fetchall()
                    col_list = [col[0] for col in columns_info]
                    row_count_query = text(f"SELECT COUNT(*) FROM `{table_name}`")
                    row_count = conn.execute(row_count_query).scalar()
                    overview[table_name] = {
                        "columns": col_list,
                        "row_count": row_count,
                        "column_count": len(col_list)
                    }
                    print(f"Processed table {table_name}: {overview[table_name]}")
                except Exception as e:
                    print(f"Error processing table {table_name}: {e}")
    except Exception as e:
        st.error(f"Error retrieving DB overview: {e}")
        print(f"Database error: {e}")
    return overview

# -----------------------------------------------------------------------------
# 3. 切换表展示状态
# -----------------------------------------------------------------------------
def toggle_table(table_name: str):
    if table_name not in st.session_state.db_overview_expanded:
        st.session_state.db_overview_expanded[table_name] = False
    st.session_state.db_overview_expanded[table_name] = not st.session_state.db_overview_expanded[table_name]

# -----------------------------------------------------------------------------
# 4. 切换列展示状态
# -----------------------------------------------------------------------------
def toggle_column(table_name: str, column_name: str):
    key = f"show_{table_name}_{column_name}"
    current = st.session_state.get(key, False)
    st.session_state[key] = not current

def handle_table_selection(table_name):
    # Store the previous selected tables to detect changes
    previous_tables = set(st.session_state.selected_tables)
    
    if st.session_state[f"selected_table_{table_name}"]:
        if table_name not in st.session_state.selected_tables and len(st.session_state.selected_tables) >= 1:
            st.warning("You cannot select more than one table.")
            st.session_state[f"selected_table_{table_name}"] = False
        else:
            st.session_state.selected_tables.add(table_name)
    else:
        st.session_state.selected_tables.discard(table_name)
    
    # If the selected tables have changed, reset keyword-related state
    if previous_tables != st.session_state.selected_tables:
        # Reset tables_confirmed flag
        st.session_state.tables_confirmed = False
        
        # Reset keyword selections
        if "selected_dt_field" in st.session_state:
            st.session_state.selected_dt_field = None
        if "selected_region_fields" in st.session_state:
            st.session_state.selected_region_fields = []
        if "keywords_confirmed" in st.session_state:
            st.session_state.keywords_confirmed = False
        
        # Also clear the dt_fields and region_fields to force re-fetching for the new table
        if "dt_fields" in st.session_state:
            del st.session_state.dt_fields
        if "region_fields" in st.session_state:
            del st.session_state.region_fields

def handle_column_selection(table_name, column_name):
    key = (table_name, column_name)
    if st.session_state[f"selected_column_{table_name}_{column_name}"]:
        st.session_state.selected_columns.add(key)
    else:
        st.session_state.selected_columns.discard(key)

# -----------------------------------------------------------------------------
# 5. 获取某列中最多5个不同的值，若超过则显示“...”
# -----------------------------------------------------------------------------
def get_top5_values(engine, table_name: str, column_name: str):
    try:
        with engine.connect() as conn:
            query = text(f"SELECT DISTINCT `{column_name}` FROM `{table_name}` LIMIT 6")
            rows = conn.execute(query).fetchall()
            values = [str(row[0]) for row in rows]
            has_more = (len(values) == 6)
            if has_more:
                values = values[:5]
            return values, has_more
    except Exception as e:
        st.error(f"Error retrieving distinct values for {table_name}.{column_name}: {e}")
        return [], False

# -----------------------------------------------------------------------------
# 6. Streamlit 界面：左右中三列布局
# -----------------------------------------------------------------------------
st.set_page_config(page_title="SSH + DB + Chat Layout", layout="wide")
col_left, col_mid, col_right = st.columns([0.5, 1, 1])

# 左侧：连接设置
with col_left:
    st.subheader("Connection Settings")
    with st.form("connection_form"):
        ssh_key = st.file_uploader("Upload your SSH key (.pem)", type=["pem"])
        ssh_address = st.text_input("SSH Server Address", 
            value=st.session_state.connection_settings["ssh_address"])
        ssh_port = st.number_input("SSH Port", 
            value=st.session_state.connection_settings["ssh_port"])
        ssh_username = st.text_input("SSH Username", 
            value=st.session_state.connection_settings["ssh_username"])
        remote_bind_address = st.text_input("Database Host", 
            value=st.session_state.connection_settings["remote_bind_address"])
        remote_bind_port = st.number_input("Database Port", 
            value=st.session_state.connection_settings["remote_bind_port"])
        db_user = st.text_input("Database User", 
            value=st.session_state.connection_settings["db_user"])
        db_password = st.text_input("Database Password", 
            value=st.session_state.connection_settings["db_password"], 
            type="password")
        db_name = st.text_input("Database Name", 
            value=st.session_state.connection_settings["db_name"])
        
        if st.form_submit_button("Connect"):
            if ssh_key is None:
                st.warning("Please upload your SSH key first.")
            else:
                with open("temp_ssh_key.pem", "wb") as f:
                    f.write(ssh_key.getvalue())
                connection_settings = {
                    "ssh_address": ssh_address,
                    "ssh_port": ssh_port,
                    "ssh_username": ssh_username,
                    "ssh_key_path": "temp_ssh_key.pem",
                    "remote_bind_address": remote_bind_address,
                    "remote_bind_port": remote_bind_port,
                    "db_user": db_user,
                    "db_password": db_password,
                    "db_name": db_name
                }
                from sqlconnect import db_connection
                if db_connection.connect(connection_settings):
                    st.session_state["engine"] = db_connection.get_engine()
                    engine = db_connection.get_engine() 
                    get_engine()  # 更新全局变量 engine
                    st.success("Engine created successfully")
                    shared.engine = engine
                    st.session_state["db_overview"] = get_db_overview(db_connection.get_engine())
                    st.success("Connected successfully!")
                    st.rerun()
                else:
                    st.error("Failed to connect to database")

# 中间：聊天界面
with col_mid:
    st.subheader("Chat Window")
    
    # Convert the set to a list before iteration to avoid "Set changed size during iteration" error
    if st.session_state.selected_tables or st.session_state.selected_columns:
        st.write("**Selected Items:**")
        html_str = "<div style='display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 15px;'>"
        # Create a copy of the set before iterating
        selected_tables_list = list(st.session_state.selected_tables)
        for table in selected_tables_list:
            table_info = st.session_state.db_overview.get(table, {})
            explanation = get_table_explanation(table, st.session_state.engine)
            col_count = table_info.get("column_count", len(table_info.get("columns", [])))
            row_count = table_info.get("row_count", 0)
            html_str += f"""
                <div title="{explanation}" 
                     style="padding: 3px 8px; 
                            background-color: #f0f2f6; 
                            border-radius: 15px; 
                            font-size: 0.8em;
                            cursor: default;">
                    {table} ({col_count}Col, {row_count}Row)
                </div>"""
        html_str += "</div>"
        st.markdown(html_str, unsafe_allow_html=True)
    
    # Track table confirmation status
    if "tables_confirmed" not in st.session_state:
        st.session_state.tables_confirmed = False
    
    if st.session_state.selected_tables and not st.session_state.tables_confirmed:
        if st.button("Confirm Table Selection"):
            try:
                confirmed = confirm_selected_tables(st.session_state.selected_tables)
                if confirmed:
                    st.session_state["confirmed_tables"] = confirmed
                    st.session_state.tables_confirmed = True
                    st.success(f"Confirmed tables: {', '.join(confirmed)}")
                    print("Backend confirmed tables:", confirmed)
                    st.rerun()  # Rerun to show keyword selection interface
                else:
                    st.error("No tables were successfully confirmed.")
            except Exception as e:
                st.error(f"Error during table confirmation: {e}")
    
    # Display keyword selection interface after table confirmation
    if st.session_state.tables_confirmed:
        # Add a section for keyword selection
        st.write("### Keyword Selection")
        st.write("Please select appropriate datetime and region keywords for your query.")
        
        # Modified keyword selection function (horizontal layout)
        # Call the modified show_keyword_selection_interface function
        keyword_selection_successful = show_keyword_selection_interface()
        
        # Add a button to confirm keyword selection
        if (st.session_state.get("selected_dt_field") and 
            st.session_state.get("selected_region_fields")):
            if "keywords_confirmed" not in st.session_state:
                st.session_state.keywords_confirmed = False
                
            if not st.session_state.keywords_confirmed:
                if st.button("Confirm Keyword Selection"):
                    st.session_state.keywords_confirmed = True
                    st.success("Keywords confirmed! You can now enter your query.")
                    st.rerun()
    
    # Display the chat messages
    chat_container = st.container(height=500)
    for message in st.session_state.messages:
        with chat_container:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Enable chat input only if keywords are confirmed or tables were not selected
    chat_input_disabled = (st.session_state.tables_confirmed and 
                          not st.session_state.get("keywords_confirmed", False))
    
    # Update the chat input field
    if prompt := st.chat_input("Type your message here...", disabled=chat_input_disabled):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Getting response..."):
            # Pass the selected keywords to the dateandregion_filter function
            ai_response = get_ai_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
        with chat_container:
            for message in st.session_state.messages[-2:]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

# 右侧：数据库概览与结构展示
from table_schema_handler import get_or_generate_table_schema

# Right column: Database Overview & AI-Generated Table Categories
with col_right:
    st.subheader("Database Overview & Structure")
    
    # Initialize db_connected flag to determine which section to show
    db_connected = (
        "engine" in st.session_state and 
        st.session_state["engine"] is not None and 
        "db_overview" in st.session_state
    )
    
    # Database statistics section
    if db_connected:
        overview_dict = st.session_state.db_overview
        total_tables = len(overview_dict)
        total_rows = sum(info["row_count"] for info in overview_dict.values())
        total_columns = sum(info["column_count"] for info in overview_dict.values())
        total_data = sum(info["row_count"] * info["column_count"] for info in overview_dict.values())
        
        # Display database statistics
        st.markdown(f"""
        **Database Statistics**  
        Total Tables: `{total_tables}` Total Data: `{total_data:,}`
        """)
        
        
        # Check if tableinfo.txt exists
        if not os.path.exists("tableinfo.txt"):
            st.error("tableinfo.txt not found. Please create this file with table descriptions.")
            st.info("The file should contain table names and descriptions in the format: 'table_name : description'")
        else:
            # Get or generate the table schema - only generate when needed
            with st.spinner("Loading table categories..."):
                # Check if we need to get/generate schema
                json_data = get_or_generate_table_schema()
            
            if json_data:
                
                # Option to regenerate the classification if needed (hidden in expander)
                with st.expander("Advanced Options", expanded=False):
                    if st.button("Regenerate Table Categories"):
                        with st.spinner("Regenerating table categories with AI..."):
                            json_data = get_or_generate_table_schema(force_regenerate=True)
                        if json_data:
                            st.success("Table categories regenerated!")
                            st.rerun()
                        else:
                            st.error("Failed to regenerate table categories.")
                
                # Search box
                search_query = st.text_input("Search in table categories", key="json_search_query_main")
                
                # JSON viewer with table selection
                json_container = st.container(height=500)
                with json_container:
                    # Search functionality
                    if search_query:
                        search_results = json_handler.search_json_data(json_data, search_query)
                        if search_results:
                            st.subheader(f"Search Results for '{search_query}'")
                            json_handler.display_json_hierarchy_with_selection(
                                search_results, 
                                table_info=overview_dict,
                                handle_table_selection=handle_table_selection,
                                toggle_table=toggle_table
                            )
                        else:
                            st.info(f"No results found for '{search_query}'")
                    else:
                        # Display full hierarchy with selection capability
                        json_handler.display_json_hierarchy_with_selection(
                            json_data, 
                            table_info=overview_dict,
                            handle_table_selection=handle_table_selection,
                            toggle_table=toggle_table
                        )
                    
                # Display expanded table details if any tables are expanded
                for table_name in st.session_state.db_overview_expanded:
                    if st.session_state.db_overview_expanded.get(table_name, False):
                        st.markdown(f"### Details for table: {table_name}")
                        info = overview_dict.get(table_name, {})
                        
                        if "columns" in info:
                            for col_name in info["columns"]:
                                col_checkbox_key = f"selected_column_{table_name}_{col_name}"
                                if col_checkbox_key not in st.session_state:
                                    st.session_state[col_checkbox_key] = False
                                
                                cols = st.columns([0.2, 0.5, 0.3])
                                with cols[0]:
                                    st.checkbox(
                                        label=f"Select column {col_name} from {table_name}",
                                        key=col_checkbox_key,
                                        on_change=handle_column_selection,
                                        args=(table_name, col_name),
                                        label_visibility="collapsed",
                                        value=st.session_state[col_checkbox_key]
                                    )
                                with cols[1]:
                                    if st.button(col_name, key=f"btn_{table_name}_{col_name}"):
                                        toggle_column(table_name, col_name)
                                with cols[2]:
                                    if st.session_state.get(f"show_{table_name}_{col_name}", False):
                                        values, has_more = get_top5_values(
                                            st.session_state["engine"],
                                            table_name,
                                            col_name
                                        )
                                        display_str = ", ".join(values) + ("..." if has_more else "")
                                        st.caption(display_str)
                            st.markdown("---")
            else:
                st.error("Failed to load or generate table categories. Please check the logs for details.")
    else:
        st.warning("Not connected to any database yet. Connect to a database to view and select tables.")