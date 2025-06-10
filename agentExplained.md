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
- `code`：生成的 TypeScript 代码块（如果成功）
- `dialog`：LLM 对话记录
- `agent`：`"ConceptDeriveAgent"`

---

### **Workflow**

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
