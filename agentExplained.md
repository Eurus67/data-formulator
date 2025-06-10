## Concept Derivation Agent

使用语言模型（例如 OpenAI GPT）来生成 TypeScript 风格的代码，该代码可以从输入表中派生出新的概念或字段。

---

###  **输入**
| Parameter       | Type         | 
|----------------|--------------|
| `input_table`   | `dict`       |
| `input_fields`  | `List[str]`  |
| `output_field`  | `str`        |
| `description`   | `str`        | 
| `n`             | `int`        | 

---

### **输出**
返回一个包含n个候选字典结果的列表，每个结果包含以下内容：
- `status`：`'ok'` 或 `'other error'`
- `content`：生成的 TypeScript 代码块（如果成功）
- `dialog`：LLM 对话记录
- `agent`：`"ConceptDeriveAgent"`

---

### **执行步骤**

#### **1. 生成数据总结**

```python
data_summary = generate_data_summary([input_table], include_data_samples=True)
```

---

#### **2. 推断`input_fields`中每个field对应的 ts 数据类型, 并生成符合ts格式的参数字符串**

```python
input_fields_info = [{"name": name, "type": infer_ts_datatype(pd.DataFrame(input_table['rows']), name)} for name in input_fields]
arg_string = ", ".join([f"{field_name_to_ts_variable_name(field['name'])} : {field['type']}" for field in input_fields_info])
```

---

#### **3. 提供生成代码的模板并构建user_query**

```python
code_template = f"```typescript\n//{description}\n({arg_string}) => {{\n    // complete code here\n    return {field_name_to_ts_variable_name(output_field)}\n}}\n```"

user_query = f"[CONTEXT]\n\n{data_summary}\n\n[GOAL]\n\n{description}\n\n[TEMPLATE]\n\n{code_template}\n\n[OUTPUT]\n"
```

---

#### **5. 模型推理**

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_query}
]
response = self.client.get_completion(messages=messages)
```

---

#### **6. 生成并返回candidates**
- 从每个回答里提取 TypeScript 代码块
- 将状态、生成的代码、对话记录、agent名字存为一个candidate
- 可以有多个candidates, 最终返回一个列表

```python
candidates = []
for choice in response.choices:
    code_blocks = extract_code_from_gpt_response(choice.message.content + "\n", "typescript")

    if len(code_blocks) > 0:
        result = {'status': 'ok', 'code': code_blocks[-1]}
    else:
        result = {'status': 'other error', 'content': 'unable to extract code from response'}

    result['dialog'] = [
        *messages,
        {"role": choice.message.role, "content": choice.message.content}
    ]
    result['agent'] = 'ConceptDeriveAgent'
    candidates.append(result)
```

---

## Py Concept Derivation Agent

根据用户对新字段的描述，调用 LLM 生成对应的数据转换代码，执行后返回包含新字段的表格

---

###  **输入**
| Parameter       | Type         | 
|----------------|--------------|
| `input_table`   | `dict`       |
| `input_fields`  | `List[str]`  |
| `output_field`  | `str`        |
| `description`   | `str`        | 

---

### **输出**
返回一个包含n个候选字典结果的列表，每个结果包含以下内容：
- `status`：`'ok'` 或 `'other error'`
- `content`：生成的包含新字段的表格（以字典列表的形式）
- `dialog`：LLM 对话记录
- `agent`：`"ConceptDeriveAgent"`

---

### **执行步骤**

#### **1. 生成数据总结**

```python
data_summary = generate_data_summary([input_table], include_data_samples=True)
```

---

#### **2. 推断`input_fields`中每个field对应的 ts 数据类型, 并生成符合ts格式的参数字符串**

```python
input_fields_info = [{"name": name, "type": infer_ts_datatype(pd.DataFrame(input_table['rows']), name)} for name in input_fields]
arg_string = ", ".join([f"{field_name_to_ts_variable_name(field['name'])} : {field['type']}" for field in input_fields_info])
```

---

#### **3. 构建user_query**

```python
objective = {
            "input_fields": input_fields,
            "output_field": output_field,
            "description": description
        }
        
user_query = f"[CONTEXT]\n\n{data_summary}\n\n[GOAL]\n\n{objective}\n\n[OUTPUT]\n"
```

---

#### **5. 模型推理**

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_query}
]
response = self.client.get_completion(messages=messages)
```

---

#### **6. 生成并返回candidates**
- 从每个回答里提取 Python代码块
- 在sandbox中运行提取的代码（run_derive_concept），生成新的pd dataframe
- 将运行状态、新dataframe、对话记录、agent名字存为一个candidate
- 可以有多个candidate, 最终返回一个candidates列表

```python
candidates = []
        for choice in response.choices:
            
            logger.info("\n=== Python Data Derive Agent ===>\n")
            logger.info(choice.message.content + "\n")

            code_blocks = extract_code_from_gpt_response(choice.message.content + "\n", "python")

            if len(code_blocks) > 0:
                code_str = code_blocks[-1]
                try:
                    result =  py_sandbox.run_derive_concept(code_str, output_field, input_table['rows'], self.exec_python_in_subprocess)

                    if result['status'] == 'ok':
                        result['content'] = {
                            'rows': result['content'].to_dict(orient='records'),
                        }
                    else:
                        print(result['content'])
                    result['code'] = code_str
                except Exception as e:
                    print('other error:')
                    error_message = traceback.format_exc()
                    print(error_message)
                    result = {'status': 'other error', 'content': error_message}
            else:
                result = {'status': 'other error', 'content': 'unable to extract code from response'}

            result['dialog'] = [*messages, {"role": choice.message.role, "content": choice.message.content}]
            result['agent'] = 'PyConceptDeriveAgent'
            candidates.append(result)
```

---

# DataCleanAgent

`DataCleanAgent` 使用LLM根据用户指定的输入推断适当的清理操作来清理原始数据

---

### **输入**

| Parameter | Type |
|----------|------|
| `content_type` | `"text"` or `"image"` |
| `raw_data` | `str` |
| `image_cleaning_instruction` | `str` | 

---

### **输出**
返回一个包含candidate列表，每个结果包含以下内容：
- `status`：`'ok'` 或 `'other error'`
- `content`：csv格式的经过清理的数据
- `dialog`：LLM 对话记录
- `agent`：'DataCleanAgent'

---

### **执行步骤**

#### **1. 根据不同输入数据类型生成user_prompt**
如果是"text", 包括输入的原数据

```python
user_prompt = {
                "role": "user",
                "content": [{
                    'type': 'text',
                    'text': f"[DATA]\n\n{raw_data}\n\n[OUTPUT]\n"
                }]
            }
```
---
如果是"image"，包括图片url和非必须的用户指令

```python
user_prompt = {
                'role': 'user',
                'content': [ {
                    'type': 'text',
                    'text': '''[RAW_DATA]\n\n'''},
                    {
                        'type': 'image_url',
                        'image_url': {
                            "url": raw_data,
                            "detail": "high"
                        }
                    },
                    {
                        'type': 'text',
                        'text': f'''{cleaning_prompt}[OUTPUT]\n\n'''
                    }, 
                ]
            }
```
---

#### **2. 模型推理**
system_message需要和user_prompt保持格式一致。

```python
system_message = {
            'role': 'system',
            'content': [ {'type': 'text', 'text': SYSTEM_PROMPT}]}

messages = [system_message, user_prompt]
        

response = self.client.get_completion(messages = messages)
```
system_prompt中的要求包括让AI生成csv数据块和一个包括以下内容的json：
- mode："data generation" or "data cleaning" （但是没用到）
- reason: 解释一下清洗原因

---

#### **3. 生成并返回candidates**
- 从每个回答里提取csv数据块和json
- 将csv、json内容、对话记录、agent名字存为一个candidate
- 可以有多个candidates, 最终返回一个列表

```python
candidates = []
for choice in response.choices:
    code_blocks = extract_code_from_gpt_response(choice.message.content + "\n", "typescript")

    if len(code_blocks) > 0:
        result = {'status': 'ok', 'code': code_blocks[-1]}
    else:
        result = {'status': 'other error', 'content': 'unable to extract code from response'}

    result['dialog'] = [
        *messages,
        {"role": choice.message.role, "content": choice.message.content}
    ]
    result['agent'] = 'ConceptDeriveAgent'
    candidates.append(result)
```

---
