
import json
import os
import re
import datasets
# from verl.utils.hdfs_io import copy, makedirs
import argparse

def read_json(file_path):
    """
    Reads and loads a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict or list: The loaded JSON data (dictionary or list).
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        print(f"Successfully loaded JSON from '{file_path}'.")
        return data
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from file '{file_path}'.")
        print(f"Details: {e}")
        return None

def print_json(data, indent=4):
    """
    Prints JSON data in a pretty format.

    Args:
        data (dict or list): The JSON data to print.
        indent (int): The number of spaces for indentation.
    """
    if data is None:
        print("No data to print.")
        return

    print(json.dumps(data, indent=indent, ensure_ascii=False))

 

rawData8kDir ='/vepfs-cnbj3fa964354bf4/zyd/gemin2volczyd/verl-main/verl-main/data/mathlv345_8k_chatml.json' # "tools_info.json"  # Replace with your JSON file path

json_data = read_json(rawData8kDir)
print(len(json_data),type(json_data))
print_json(json_data[1])
num_few_shot = 5
data_source = 'openai/gsm8k'

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str) # extract the solution after ####
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution
#<|im_start|>system\nYou are a helpful assistant good at solving math problems with step-by-step reasoning. You should first think about the reasoning process in your mind and then provide the user with the answer. Your answer must be in LaTeX format and wrapped in $...$. The reasoning process and answer are enclosed within <think> and </think> tags, respectively, i.e., <think>Since $1+1=2$, so the answer is $2$.</think><answer>$2$</answer>.
            # Additionally, you can use A Python_Code_Generator_Tool that generates and executes simple Python code snippets for basic arithmetical calculations and math-related problems. The generated code runs in a highly restricted environment with only basic mathematical operations available.
            # input_types={"Tool input": "str - A clear, specific description of the arithmetic calculation or math problem to be solved, including any necessary numerical inputs."},
            # output_type="dict - A dictionary containing the generated code, calculation result, and any error messages.put the "Tool input" between <code> and <code>\n
            # <|im_end|>\n
formatData=[]
for idx,item in enumerate(json_data):
    #instruction_following = "Let's think step by step and output the final answer after \"####\"."
    question = item["prompt"]
    pattern = r"<\|im_start\|>user(.*?)<\|im_end\|>"
    match = re.search(pattern, question, re.DOTALL)
    if match:
        question = match.group(1).strip() 
    print(question)
    #question = question + ' ' + instruction_following
    answer = item["answer"]
    #solution = extract_solution(answer)
    temp = [{
        "data_source": "math345_8k",
        "prompt": [{
            "role": "user",
            "content": '''You are a helpful assistant skilled at solving math problems with step-by-step reasoning. Your responses must follow these guidelines:
Reasoning Process : Begin by explaining the problem-solving process in detail within <think> tags. Use LaTeX for mathematical expressions, wrapped in $...$.
Final Answer : Provide the final answer within <answer> tags, also formatted in LaTeX.
Python Verification : Include Python code to verify the computation. The code should be enclosed within <code> tags. Ensure the result of the verification is stored in a variable named result and printed.
Format Example : Below is an example to illustrate the expected structure:
Example Problem : Calculate the sum of 3 and 5.
Solution : <think>
To calculate the sum of 3 and 5, we simply add the two numbers together:
$3 + 5 = 8$
So the answer is $8$.
</think>
<answer>$8$</answer>
<code>
# Python code to verify the computation
result = 3 + 5
print(result)
</code>
Ensure your response is clear, concise, and adheres to this structure.Now anwer the question: '''+question #+' '+"Please integrate natural language reasoning with programs to solve the problem above"
        }],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer
        },
        "extra_info": {
            'split': 'train',
            'index': idx
        }
    }]
    formatData.extend(temp)
    #break
print(len(formatData),type(formatData))
local_dir='/vepfs-cnbj3fa964354bf4/zyd/gemin2volczyd/verl-main/verl-main/data/math345_8k_code.parquet'
import pandas as pd
# 将 JSON 数据加载到 Pandas DataFrame
df = pd.DataFrame(formatData)

# 将 DataFrame 保存为 Parquet 文件
df.to_parquet(local_dir, engine="pyarrow")

print(f"JSON 数据已成功转换为 Parquet 格式并保存到 '{local_dir}' 文件中。")
#test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
#test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
