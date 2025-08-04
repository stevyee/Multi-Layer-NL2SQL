# from langchain_openai import ChatOpenAI
# import getpass
# import os
# from dotenv import load_dotenv
# from langchain_deepseek import ChatDeepSeek
# from openai import OpenAI
# from chain import llmo1
# import shared
# import re
# import subprocess
# import pandas as pd
# import tempfile
# from table_selection_handler import get_current_df

# # 載入 .env 檔案中的環境變數
# load_dotenv()
# # if not os.getenv("DEEPSEEK_API_KEY"):
# #     os.environ["DEEPSEEK_API_KEY"] = getpass.getpass()
# # # 使用正確的模型名稱，並從環境變數獲取 API 金鑰

# # llm = ChatDeepSeek(model="deepseek-reasoner", temperature=0)  


# client = OpenAI(api_key="sk-07a75ef9e33146ed84f15e0fd621baf4", base_url="https://api.deepseek.com")


# # response = client.chat.completions.create(
# #         model="deepseek-chat",
# #         messages=[
# #             {"role": "system", "content": "You are a helpful assistant"},
# #             {"role": "user", "content": "Hello"},
# #         ],


# # 假设 llmo1 已经初始化为你的 LLM 调用对象
# # 例如：llmo1 = ChatOpenAI(model='o1-mini', temperature=1)
# # 另外 car_df 是你选定的 DataFrame

# def save_table_to_csv(df, filename):
#     """Save DataFrame to CSV if it exists"""
#     if df is not None and isinstance(df, pd.DataFrame):
#         try:
#             df.to_csv(filename, index=False)
#             return True
#         except Exception as e:
#             print(f"Error saving to CSV: {e}")
#             return False
#     return False

# # def compile_and_run(code: str) -> str:
# #     """
# #     通过调用 Flask 执行器 API 来运行生成的代码，并返回执行结果
# #     """
# #     url = "http://localhost:5001/execute"  # 执行器地址
# #     try:
# #         response = requests.post(url, json={"code": code}, timeout=10)
# #         if response.status_code == 200:
# #             return response.json().get("result", "")
# #         else:
# #             return f"执行错误：{response.text}"
# #     except Exception as e:
# #         return f"请求执行器时出错：{str(e)}"

# def compile_and_run(code: str) -> str:
    
#     with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as tmp:
#         tmp.write(code)
#         tmp.flush()
#         temp_file_path = tmp.name

#     try:
#         result = subprocess.run(
#             ['python', temp_file_path],
#             capture_output=True,
#             text=True,
#             timeout=10
#         )
#         # 如果 stdout 为空，返回 stderr 以便调试错误


#         return result.stdout if result.stdout.strip() else result.stderr
#     except Exception as e:
#         return f"代码执行出错：{str(e)}"

# # 创建一个临时文件，写入代码，并获取文件路径
# tablename = get_table_description_only_datetime

# def response(user_input):
#     table = get_current_df()  # 本地 DataFrame
#     llmo1 = ChatOpenAI(model='o1-mini', temperature=1)
#     # 先将 table 保存为 CSV 文件，让生成的代码可以加载它
#     save_table_to_csv(table, "data.csv")
#     print(table)
#     print(tablename)
#     extracted_col = tablename[0]
#     code_generation_prompt = f"""
# 任务：根据以下描述生成 Python 代码来解决用户关于 pandas DataFrame 的问题，请在代码中使用 pandas。请确保代码是可运行的，并在末尾输出结果。
# 要求：
# 1. 必须使用整个 DataFrame，不允许随机抽样。
# 2. 请通过 pandas 从 CSV 文件 'data.csv' 导入数据，并赋值给变量 df。
# 3. 请判断用户的问题是否与时间有关，如果是，用户问题的描述更正为
# ’{user_input}+，请使用{extracted_col}为关键词‘，并且在你生成的python代码中加上：print("使用了‘{extracted_col}’为提取栏")
# 否则：问题描述：
# {user_input}

#         """
#     generated_response = llmo1.invoke(code_generation_prompt)
#     response_content = generated_response.content
        
#         # 打印 agent 生成的完整响应（方便调试）
#     print("agent生成的完整响应：")
#         #print(response_content)
        
#         # 尝试提取 markdown 格式的代码块
#     code_match = re.search(r"```python(.*?)```", response_content, re.DOTALL)
#     if code_match:
#             code = code_match.group(1).strip()
#     else:
#             code = response_content.strip()
        
#         # 打印提取出的代码（方便调试）
#     print("提取的代码：")
#     with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as tmp:
#             tmp.write(code)
#             tmp.flush()  # 确保数据写入文件
#             temp_file_path = tmp.name

#     print("生成的临时文件路径：", temp_file_path)
        
#         # 调用执行器执行生成的代码
#     execution_result = compile_and_run(code)
#     result = execution_result.splitlines()
#     analysis_prompt = f"you are given this user query: {user_input} and this is the result: {result}, please give a conclusion and implement analysis about this result for future decision making in chinese"
#     analysis = llmo1.invoke(analysis_prompt)   
#     print(analysis.content)
#     return analysis.content

# __all__ = ['response']