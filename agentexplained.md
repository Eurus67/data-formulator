### Agent: `SQLDataTransformationAgent`

**Summary**:  
This agent interfaces with a language model to generate, execute, and refine SQL queries on DuckDB tables based on natural language instructions.

**Inputs**:
- `input_tables`: list of dicts (each containing `name` and `rows`)
- `description`: string description of the desired transformation
- `expected_fields`: list of strings for output fields
- `prev_messages`: prior message history (optional)
- `dialog`: chat history for follow-up (in `followup`)
- `new_instruction`: refinement instruction (in `followup`)
- `client`: LLM client for prompt completion
- `conn`: DuckDB connection
- `system_prompt`: optional override of default system prompt

**Outputs**:
- Returns a list of result dicts, each containing SQL code, data rows, metadata, and refined goals

---

### Function: `__init__`

**Step**: Initialize agent with client, DuckDB connection, and optional prompt

```python
def __init__(self, client, conn, system_prompt=None):
    self.client = client
    self.conn = conn
    self.system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
```

---

### Function: `process_gpt_sql_response`

**Step 1**: Handle response errors

```python
if isinstance(response, Exception):
    result = {'status': 'other error', 'content': str(response.body)}
    return [result]
```

**Step 2**: Loop through model choices, extract refined goal and SQL

```python
for choice in response.choices:
    logger.info("=== SQL query result ===>")
    logger.info(choice.message.content + "\n")

    json_blocks = extract_json_objects(choice.message.content + "\n")
    if len(json_blocks) > 0:
        refined_goal = json_blocks[0]
    else:
        refined_goal = {'visualization_fields': [], 'instruction': '', 'reason': ''}

    query_blocks = extract_code_from_gpt_response(choice.message.content + "\n", "sql")
```

**Step 3**: Execute SQL, limit rows, handle success/error

```python
if len(query_blocks) > 0:
    query_str = query_blocks[-1]
    try:
        random_suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
        table_name = f"view_{random_suffix}"
        create_query = f"CREATE VIEW IF NOT EXISTS {table_name} AS {query_str}"
        self.conn.execute(create_query)
        self.conn.commit()
        row_count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        if row_count > 5000:
            query_output = self.conn.execute(f"SELECT * FROM {table_name} LIMIT 5000").fetch_df()
        else:
            query_output = self.conn.execute(f"SELECT * FROM {table_name}").fetch_df()

        result = {
            "status": "ok",
            "code": query_str,
            "content": {
                'rows': json.loads(query_output.to_json(orient='records')),
                'virtual': {
                    'table_name': table_name,
                    'row_count': row_count
                }
            },
        }

    except Exception as e:
        logger.warning('Error occurred during code execution:')
        error_message = f"An error occurred during code execution. Error type: {type(e).__name__}"
        logger.warning(error_message)
        result = {'status': 'error', 'code': query_str, 'content': error_message}
```

**Step 4**: Handle missing SQL and append result

```python
else:
    result = {'status': 'error', 'code': "", 'content': "No code block found in the response. The model is unable to generate code to complete the task."}

result['dialog'] = [*messages, {"role": choice.message.role, "content": choice.message.content}]
result['agent'] = 'SQLDataTransformationAgent'
result['refined_goal'] = refined_goal
candidates.append(result)
```

**Step 5**: Log final candidates and return

```python
for candidate in candidates:
    for key, value in candidate.items():
        if key in ['dialog', 'content']:
            logger.info(f"##{key}:\n{str(value)[:1000]}...")
        else:
            logger.info(f"## {key}:\n{value}")

return candidates
```

---

### Function: `run`

**Step 1**: Register input tables if not already in DuckDB

```python
for table in input_tables:
    table_name = sanitize_table_name(table['name'])
    try:
        self.conn.execute(f"DESCRIBE {table_name}")
    except Exception:
        df = pd.DataFrame(table['rows'])
        self.conn.register(f'df_temp', df)
        self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df_temp")
        self.conn.execute(f"DROP VIEW df_temp")
        r = self.conn.execute(f"SELECT * FROM {table_name} LIMIT 10").fetch_df()
        print(r)
        logger.info(f"Created table {table_name} from dataframe")
```

**Step 2**: Format previous messages if any

```python
if len(prev_messages) > 0:
    logger.info("=== Previous messages ===>")
    formatted_prev_messages = ""
    for m in prev_messages:
        if m['role'] != 'system':
            formatted_prev_messages += f"{m['role']}: \n\n\t{m['content']}\n\n"
    logger.info(formatted_prev_messages)
    prev_messages = [{"role": "user", "content": '[Previous Messages] Here are the previous messages for your reference:\n\n' + formatted_prev_messages}]
```

**Step 3**: Generate data summary for prompt context

```python
data_summary = ""
for table in input_tables:
    table_name = sanitize_table_name(table['name'])
    table_summary_str = get_sql_table_statistics_str(self.conn, table_name)
    data_summary += f"[TABLE {table_name}]\n\n{table_summary_str}\n\n"
```

**Step 4**: Format user query and send to LLM

```python
goal = {
    "instruction": description,
    "visualization_fields": expected_fields
}

user_query = f"[CONTEXT]\n\n{data_summary}[GOAL]\n\n{json.dumps(goal, indent=4)}\n\n[OUTPUT]\n"
logger.info(user_query)

messages = [{"role":"system", "content": self.system_prompt},
            *prev_messages,
            {"role":"user","content": user_query}]

response = self.client.get_completion(messages = messages)
return self.process_gpt_sql_response(response, messages)
```

---

### Function: `followup`

**Step**: Modify previous dialog and request updated SQL

```python
goal = {
    "followup_instruction": new_instruction,
    "visualization_fields": output_fields
}

logger.info(f"GOAL: \n\n{goal}")

updated_dialog = [{"role":"system", "content": self.system_prompt}, *dialog[1:]]

messages = [*updated_dialog, {"role":"user", 
                      "content": f"Update the sql query above based on the following instruction:\n\n{json.dumps(goal, indent=4)}"}]

response = self.client.get_completion(messages = messages)

return self.process_gpt_sql_response(response, messages)
```

---

### Function: `get_sql_table_statistics_str`

**Step 1**: Get table columns and sample data

```python
columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
sample_data = conn.execute(f"SELECT * FROM {table_name} LIMIT {row_sample_size}").fetchall()
```

**Step 2**: Format sample data

```python
col_names = [col[0] for col in columns]
formatted_sample_data = "| " + " | ".join(col_names) + " |\n"
for i, row in enumerate(sample_data):
    formatted_sample_data += f"{i}| " + " | ".join(str(val)[:max_val_chars]+ "..." if len(str(val)) > max_val_chars else str(val) for val in row) + " |\n"
```

**Step 3**: Generate per-column metadata

```python
col_metadata_list = []
for col in columns:
    col_name = col[0]
    col_type = col[1]
    quoted_col_name = f'"{col_name}"'

    if col_type in ['INTEGER', 'DOUBLE', 'DECIMAL']:
        stats_query = f"""
        SELECT 
            COUNT(*) as count,
            COUNT(DISTINCT {quoted_col_name}) as unique_count,
            COUNT(*) - COUNT({quoted_col_name}) as null_count,
            MIN({quoted_col_name}) as min_value,
            MAX({quoted_col_name}) as max_value,
            AVG({quoted_col_name}) as avg_value
        FROM {table_name}
        """
    else:
        stats_query = f"""
        SELECT 
            COUNT(*) as count,
            COUNT(DISTINCT {quoted_col_name}) as unique_count,
            COUNT(*) - COUNT({quoted_col_name}) as null_count
        FROM {table_name}
        """

    col_stats = conn.execute(stats_query).fetchone()
```

**Step 4**: Format column statistics

```python
    if col_type in ['INTEGER', 'DOUBLE', 'DECIMAL']:
        stats_dict = dict(zip(
            ["count", "unique_count", "null_count", "min", "max", "avg"],
            col_stats
        ))
    else:
        stats_dict = dict(zip(
            ["count", "unique_count", "null_count"],
            col_stats
        ))
        query_for_sample_values = f"""
        (SELECT DISTINCT {quoted_col_name}
            FROM {table_name} 
            WHERE {quoted_col_name} IS NOT NULL 
            LIMIT {field_sample_size})
        """
        sample_values = conn.execute(query_for_sample_values).fetchall()
        stats_dict['sample_values'] = [str(val)[:max_val_chars]+ "..." if len(str(val)) > max_val_chars else str(val) for val in sample_values]

    col_metadata_list.append({
        "column": col_name,
        "type": col_type,
        "statistics": stats_dict,
    })
```

**Step 5**: Combine and return table summary string

```python
table_metadata = {
    "column_metadata": col_metadata_list,
    "sample_data_str": formatted_sample_data
}

table_summary_str = f"Column metadata:\n\n"
for col_metadata in table_metadata['column_metadata']:
    table_summary_str += f"\t{col_metadata['column']} ({col_metadata['type']}) ---- {col_metadata['statistics']}\n"
table_summary_str += f"\n\nSample data:\n\n{table_metadata['sample_data_str']}\n"

return table_summary_str
```

### Agent: `PythonDataTransformationAgent`

**Summary**:  
This agent communicates with a language model to generate and execute Python code that transforms tabular data (pandas DataFrames) based on user instructions.

**Inputs**:
- `input_tables`: list of dicts (each with `name` and `rows`)
- `description`: transformation instructions in natural language
- `expected_fields`: list of expected output fields
- `prev_messages`: prior dialogue context (optional)
- `dialog`: conversation history used in follow-ups
- `new_instruction`: refinement or extension instruction
- `client`: LLM client to generate code
- `exec_python_in_subprocess`: whether to execute the code in a subprocess

**Outputs**:
- A list of candidate result dicts including the executed code, transformed data, dialog trace, and refined task goal

---

### Function: `__init__`

**Step**: Initialize the agent with client and optional config

```python
def __init__(self, client, system_prompt=None, exec_python_in_subprocess=False):
    self.client = client
    self.system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    self.exec_python_in_subprocess = exec_python_in_subprocess
```

---

### Function: `process_gpt_response`

**Step 1**: Handle LLM failure

```python
if isinstance(response, Exception):
    result = {'status': 'other error', 'content': str(response.body)}
    return [result]
```

**Step 2**: Iterate through response choices, extract refined goal and Python code

```python
for choice in response.choices:
    logger.info("=== Data transformation result ===>")
    logger.info(choice.message.content + "\n")
    
    json_blocks = extract_json_objects(choice.message.content + "\n")
    if len(json_blocks) > 0:
        refined_goal = json_blocks[0]
    else:
        refined_goal = {'visualization_fields': [], 'instruction': '', 'reason': ''}

    code_blocks = extract_code_from_gpt_response(choice.message.content + "\n", "python")
```

**Step 3**: Run extracted Python code on the input data and parse result

```python
if len(code_blocks) > 0:
    code_str = code_blocks[-1]
    try:
        result = py_sandbox.run_transform_in_sandbox2020(
            code_str,
            [pd.DataFrame.from_records(t['rows']) for t in input_tables],
            self.exec_python_in_subprocess
        )
        result['code'] = code_str

        if result['status'] == 'ok':
            result_df = result['content']
            result['content'] = {
                'rows': json.loads(result_df.to_json(orient='records')),
            }
        else:
            logger.info(result['content'])
```

**Step 4**: Handle runtime error or missing code block

```python
    except Exception as e:
        logger.warning('Error occurred during code execution:')
        error_message = f"An error occurred during code execution. Error type: {type(e).__name__}"
        logger.warning(error_message)
        result = {'status': 'error', 'code': code_str, 'content': error_message}
else:
    result = {'status': 'error', 'code': "", 'content': "No code block found in the response. The model is unable to generate code to complete the task."}
```

**Step 5**: Finalize result with dialog and agent metadata

```python
result['dialog'] = [*messages, {"role": choice.message.role, "content": choice.message.content}]
result['agent'] = 'PythonDataTransformationAgent'
result['refined_goal'] = refined_goal
candidates.append(result)
```

**Step 6**: Log and return candidates

```python
for candidate in candidates:
    for key, value in candidate.items():
        if key in ['dialog', 'content']:
            logger.info(f"##{key}:\n{str(value)[:1000]}...")
        else:
            logger.info(f"## {key}:\n{value}")

return candidates
```

---

### Function: `run`

**Step 1**: Format previous messages for chat context

```python
if len(prev_messages) > 0:
    logger.info("=== Previous messages ===>")
    formatted_prev_messages = ""
    for m in prev_messages:
        if m['role'] != 'system':
            formatted_prev_messages += f"{m['role']}: \n\n\t{m['content']}\n\n"
    logger.info(formatted_prev_messages)
    prev_messages = [{"role": "user", "content": '[Previous Messages] Here are the previous messages for your reference:\n\n' + formatted_prev_messages}]
```

**Step 2**: Generate summary from input tables

```python
data_summary = generate_data_summary(input_tables, include_data_samples=True)
```

**Step 3**: Construct prompt and send to LLM

```python
goal = {
    "instruction": description,
    "visualization_fields": expected_fields
}

user_query = f"[CONTEXT]\n\n{data_summary}\n\n[GOAL]\n\n{json.dumps(goal, indent=4)}\n\n[OUTPUT]\n"

logger.info(user_query)

messages = [{"role":"system", "content": self.system_prompt},
            *prev_messages,
            {"role":"user","content": user_query}]

response = self.client.get_completion(messages = messages)
return self.process_gpt_response(input_tables, messages, response)
```

---

### Function: `followup`

**Step**: Extend or modify previous transformation via follow-up prompt

```python
goal = {
    "followup_instruction": new_instruction,
    "visualization_fields": output_fields
}

logger.info(f"GOAL: \n\n{goal}")

updated_dialog = [{"role":"system", "content": self.system_prompt}, *dialog[1:]]

messages = [*updated_dialog, {"role":"user", 
                      "content": f"Update the code above based on the following instruction:\n\n{json.dumps(goal, indent=4)}"}]

response = self.client.get_completion(messages = messages)

return self.process_gpt_response(input_tables, messages, response)
```

### Agent: `SQLDataRecAgent`

**Summary**:  
This agent uses a language model to recommend SQL-based data transformations or analyses on DuckDB tables based on natural language queries, executes the resulting SQL, and returns the output data and metadata.

**Inputs**:
- `input_tables`: list of dicts with `'name'` and `'rows'`
- `description`: user prompt describing what kind of recommendation is desired
- `dialog`: prior conversation history (for follow-up requests)
- `client`: LLM interface
- `conn`: DuckDB connection object
- `system_prompt`: optional prompt prepended to LLM messages

**Outputs**:
- A list of candidate responses with execution status, SQL code, output data, agent metadata, and model-refined goals

---

### Function: `__init__`

**Step**: Initialize the agent with LLM client, DuckDB connection, and optional system prompt

```python
def __init__(self, client, conn, system_prompt=None):
    self.client = client
    self.conn = conn
    self.system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
```

---

### Function: `process_gpt_response`

**Step 1**: Return early if response is an error

```python
if isinstance(response, Exception):
    result = {'status': 'other error', 'content': str(response.body)}
    return [result]
```

**Step 2**: Iterate through model-generated choices

```python
for choice in response.choices:
    logger.info("\n=== Data recommendation result ===>\n")
    logger.info(choice.message.content + "\n")

    json_blocks = extract_json_objects(choice.message.content + "\n")
    if len(json_blocks) > 0:
        refined_goal = json_blocks[0]
    else:
        refined_goal = { 'mode': "", 'recommendation': "", 'output_fields': [], 'visualization_fields': [], }

    code_blocks = extract_code_from_gpt_response(choice.message.content + "\n", "sql")
```

**Step 3**: Run SQL code if found and wrap result

```python
if len(code_blocks) > 0:
    code_str = code_blocks[-1]
    try:
        random_suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
        table_name = f"view_{random_suffix}"
        create_query = f"CREATE VIEW IF NOT EXISTS {table_name} AS {code_str}"
        self.conn.execute(create_query)
        self.conn.commit()
        row_count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        query_output = self.conn.execute(f"SELECT * FROM {table_name} LIMIT 5000").fetch_df()

        result = {
            "status": "ok",
            "code": code_str,
            "content": {
                'rows': json.loads(query_output.to_json(orient='records')),
                'virtual': {
                    'table_name': table_name,
                    'row_count': row_count
                }
            },
        }
```

**Step 4**: Handle SQL execution error

```python
    except Exception as e:
        logger.warning('other error:')
        error_message = traceback.format_exc()
        logger.warning(error_message)
        result = {'status': 'other error', 'code': code_str, 'content': f"Unexpected error: {error_message}"}
```

**Step 5**: Handle missing code block

```python
else:
    result = {'status': 'error', 'code': "", 'content': "No code block found in the response. The model is unable to generate code to complete the task."}
```

**Step 6**: Finalize and log each candidate

```python
result['dialog'] = [*messages, {"role": choice.message.role, "content": choice.message.content}]
result['agent'] = 'SQLDataRecAgent'
result['refined_goal'] = refined_goal
candidates.append(result)

logger.info("=== Recommendation Candidates ===>")
for candidate in candidates:
    for key, value in candidate.items():
        if key in ['dialog', 'content']:
            logger.info(f"##{key}:\n{str(value)[:1000]}...")
        else:
            logger.info(f"## {key}:\n{value}")

return candidates
```

---

### Function: `run`

**Step 1**: Generate table statistics for each input table

```python
data_summary = ""
for table in input_tables:
    table_name = sanitize_table_name(table['name'])
    table_summary_str = get_sql_table_statistics_str(self.conn, table_name)
    data_summary += f"[TABLE {table_name}]\n\n{table_summary_str}\n\n"
```

**Step 2**: Construct prompt and send to LLM

```python
user_query = f"[CONTEXT]\n\n{data_summary}\n\n[GOAL]\n\n{description}\n\n[OUTPUT]\n"
logger.info(user_query)

messages = [{"role":"system", "content": self.system_prompt},
            {"role":"user","content": user_query}]

response = self.client.get_completion(messages = messages)

return self.process_gpt_response(input_tables, messages, response)
```

---

### Function: `followup`

**Step**: Append new instruction to the existing conversation and re-query the LLM

```python
logger.info(f"GOAL: \n\n{new_instruction}")

messages = [*dialog, {"role":"user", "content": f"Update: \n\n{new_instruction}"}]

response = self.client.get_completion(messages = messages)

return self.process_gpt_response(input_tables, messages, response)
```
### Agent: `PythonDataRecAgent`

**Summary**:  
This agent uses a language model to recommend Python-based data transformations on tabular input (pandas DataFrames), executes the resulting code, and returns the transformed data along with metadata and interpretation.

**Inputs**:
- `input_tables`: list of tables (each a dict with `name` and `rows`)
- `description`: user’s natural language goal for data recommendation
- `dialog`: previous conversation history for follow-ups
- `client`: LLM client for prompt-response interaction
- `system_prompt`: optional system message prepended to all prompts
- `exec_python_in_subprocess`: whether to run code in an isolated subprocess

**Outputs**:
- A list of candidates, each including:
  - status (`ok`, `error`, or `other error`)
  - executed Python code
  - rows of output data
  - dialog context
  - metadata including refined goal and agent name

---

### Function: `__init__`

**Step**: Store LLM client, prompt, and execution config

```python
def __init__(self, client, system_prompt=None, exec_python_in_subprocess=False):
    self.client = client
    self.system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    self.exec_python_in_subprocess = exec_python_in_subprocess
```

---

### Function: `process_gpt_response`

**Step 1**: Return early if LLM raised an error

```python
if isinstance(response, Exception):
    result = {'status': 'other error', 'content': str(response.body)}
    return [result]
```

**Step 2**: Loop through LLM completions and extract goals/code

```python
for choice in response.choices:
    logger.info("\n=== Data recommendation result ===>\n")
    logger.info(choice.message.content + "\n")

    json_blocks = extract_json_objects(choice.message.content + "\n")
    if len(json_blocks) > 0:
        refined_goal = json_blocks[0]
    else:
        refined_goal = { 'mode': "", 'recommendation': "", 'output_fields': [], 'visualization_fields': [], }

    code_blocks = extract_code_from_gpt_response(choice.message.content + "\n", "python")
```

**Step 3**: Execute the Python code block using input tables

```python
if len(code_blocks) > 0:
    code_str = code_blocks[-1]
    try:
        result = py_sandbox.run_transform_in_sandbox2020(
            code_str,
            [pd.DataFrame.from_records(t['rows']) for t in input_tables],
            self.exec_python_in_subprocess
        )
        result['code'] = code_str
```

**Step 4**: Format output rows if code runs successfully

```python
        if result['status'] == 'ok':
            result_df = result['content']
            result['content'] = {
                'rows': json.loads(result_df.to_json(orient='records')),
            }
        else:
            logger.info(result['content'])
```

**Step 5**: Handle execution failure or missing code

```python
    except Exception as e:
        logger.warning('other error:')
        error_message = traceback.format_exc()
        logger.warning(error_message)
        result = {'status': 'other error', 'code': code_str, 'content': f"Unexpected error executing the code, please try again."}
else:
    result = {'status': 'error', 'code': "", 'content': "No code block found in the response. The model is unable to generate code to complete the task."}
```

**Step 6**: Append dialog and metadata, add to candidates

```python
result['dialog'] = [*messages, {"role": choice.message.role, "content": choice.message.content}]
result['agent'] = 'PythonDataRecAgent'
result['refined_goal'] = refined_goal
candidates.append(result)
```

**Step 7**: Log final candidates and return

```python
logger.info("=== Recommendation Candidates ===>")
for candidate in candidates:
    for key, value in candidate.items():
        if key in ['dialog', 'content']:
            logger.info(f"##{key}:\n{str(value)[:1000]}...")
        else:
            logger.info(f"## {key}:\n{value}")

return candidates
```

---

### Function: `run`

**Step 1**: Generate data summary for all input tables

```python
data_summary = generate_data_summary(input_tables, include_data_samples=True)
```

**Step 2**: Construct prompt and query the LLM

```python
user_query = f"[CONTEXT]\n\n{data_summary}\n\n[GOAL]\n\n{description}\n\n[OUTPUT]\n"

logger.info(user_query)

messages = [{"role":"system", "content": self.system_prompt},
            {"role":"user","content": user_query}]

response = self.client.get_completion(messages = messages)

return self.process_gpt_response(input_tables, messages, response)
```

---

### Function: `followup`

**Step**: Extend the original prompt by sending a follow-up instruction

```python
logger.info(f"GOAL: \n\n{new_instruction}")

messages = [*dialog, {"role":"user", "content": f"Update: \n\n{new_instruction}"}]

response = self.client.get_completion(messages = messages)

return self.process_gpt_response(input_tables, messages, response)
```
### Agent: `DataLoadAgent`

**Summary**:  
This agent uses a language model to interpret the structure and contents of a single table (virtual or in-memory), aiming to generate metadata or visualization-ready representations such as VegaLite scripts.

**Inputs**:
- `input_data`: a dict representing a single table, containing:
  - `'name'`: table name
  - `'rows'`: list of records (if not virtual)
  - `'virtual'`: boolean flag indicating if the table is pre-loaded in DuckDB
- `client`: LLM client to process the prompt
- `conn`: DuckDB connection (used for virtual table stats)

**Outputs**:
- A list of candidate responses, each containing:
  - status (`ok`, `other error`)
  - extracted JSON content (e.g., VegaLite spec)
  - associated dialog and agent metadata

---

### Function: `__init__`

**Step**: Store the LLM client and DuckDB connection

```python
def __init__(self, client, conn):
    self.client = client
    self.conn = conn
```

---

### Function: `run`

**Step 1**: Generate table summary depending on whether it's virtual or in-memory

```python
if input_data['virtual']:
    table_name = sanitize_table_name(input_data['name'])
    table_summary_str = get_sql_table_statistics_str(self.conn, table_name, row_sample_size=5, field_sample_size=30)
    data_summary = f"[TABLE {table_name}]\n\n{table_summary_str}"
else:
    data_summary = generate_data_summary([input_data], include_data_samples=True, field_sample_size=30)
```

**Step 2**: Create prompt and send to LLM

```python
user_query = f"[DATA]\n\n{data_summary}\n\n[OUTPUT]"

logger.info(user_query)

messages = [{"role":"system", "content": SYSTEM_PROMPT},
            {"role":"user","content": user_query}]

response = self.client.get_completion(messages = messages)
```

**Step 3**: Parse each completion into a result candidate

```python
candidates = []
for choice in response.choices:
    logger.info("\n=== Data load result ===>\n")
    logger.info(choice.message.content + "\n")

    json_blocks = extract_json_objects(choice.message.content + "\n")
    logger.info(json_blocks)

    if len(json_blocks) > 0:
        result = {'status': 'ok', 'content': json_blocks[0]}
```

**Step 4**: Try to parse fallback JSON if `extract_json_objects` failed

```python
    else:
        try:
            json_block = json.loads(choice.message.content + "\n")
            result = {'status': 'ok', 'content': json_block}
        except:
            result = {'status': 'other error', 'content': 'unable to extract VegaLite script from response'}
```

**Step 5**: Attach dialog and agent name, then return all candidates

```python
    result['dialog'] = [*messages, {"role": choice.message.role, "content": choice.message.content}]
    result['agent'] = 'DataLoadAgent'

    candidates.append(result)

return candidates
```
### Agent: `CodeExplanationAgent`

**Summary**:  
This agent takes in input tables and a transformation script (Python code), and prompts a language model to explain the purpose and logic of the code in the context of the given data.

**Inputs**:
- `input_tables`: list of tables (each with `'name'` and `'rows'`)
- `code`: a Python code string that performs data transformation
- `client`: LLM interface used to generate the explanation

**Outputs**:
- A single string containing the model-generated explanation of the code

---

### Function: `__init__`

**Step**: Store the LLM client

```python
def __init__(self, client):
    self.client = client
```

---

### Function: `run`

**Step 1**: Summarize the input data for context

```python
data_summary = generate_data_summary(input_tables, include_data_samples=True)
```

**Step 2**: Format the user prompt by including the data summary and transformation code

```python
user_query = f"[CONTEXT]\n\n{data_summary}\n\n[CODE]\n\here is the transformation code: {code}\n\n[EXPLANATION]\n"
logger.info(user_query)
```

**Step 3**: Send the prompt to the language model

```python
messages = [{"role":"system", "content": SYSTEM_PROMPT},
            {"role":"user","content": user_query}]
        
response = self.client.get_completion(messages = messages)
```

**Step 4**: Log and return the model's explanation

```python
logger.info(f"=== explanation output ===>\n{response.choices[0].message.content}\n")
return response.choices[0].message.content
```
### Agent: `SortDataAgent`

**Summary**:  
This agent receives a list of values under a named field and queries a language model to return a sorted version of the data, along with any structured metadata such as ordering explanations or transformed formats.

**Inputs**:
- `name`: string label for the list of values
- `values`: list of values to be sorted
- `client`: LLM interface for generating the response

**Outputs**:
- A list of candidate responses, each containing:
  - `status`: `ok` or `other error`
  - `content`: sorted data or metadata returned by the model
  - `dialog`: message history used for this query
  - `agent`: `"SortDataAgent"`

---

### Function: `__init__`

**Step**: Save the language model client

```python
def __init__(self, client):
    self.client = client
```

---

### Function: `run`

**Step 1**: Format the input object and construct the query

```python
input_obj = {
    'name': name,
    'value': values
}

user_query = f"[INPUT]\n\n{json.dumps(input_obj)}\n\n[OUTPUT]"
logger.info(user_query)
```

**Step 2**: Send the input to the language model

```python
messages = [{"role":"system", "content": SYSTEM_PROMPT},
            {"role":"user","content": user_query}]

response = self.client.get_completion(messages = messages)
```

**Step 3**: Parse and store each model-generated choice

```python
candidates = []
for choice in response.choices:
    logger.info("\n=== Sort data agent ===>\n")
    logger.info(choice.message.content + "\n")

    json_blocks = extract_json_objects(choice.message.content + "\n")

    if len(json_blocks) > 0:
        result = {'status': 'ok', 'content': json_blocks[0]}
```

**Step 4**: Fallback parsing if no explicit JSON object was found

```python
    else:
        try:
            json_block = json.loads(choice.message.content + "\n")
            result = {'status': 'ok', 'content': json_block}
        except:
            result = {'status': 'other error', 'content': 'unable to extract VegaLite script from response'}
```

**Step 5**: Attach dialog and agent metadata to result

```python
    result['dialog'] = [*messages, {"role": choice.message.role, "content": choice.message.content}]
    result['agent'] = 'SortDataAgent'

    candidates.append(result)

return candidates
```

