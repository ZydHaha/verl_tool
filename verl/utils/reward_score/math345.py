import re
import json
import os
import random
from mathruler.grader import extract_boxed_content, grade_answer
# import Levenshtein
# from latex2sympy2_extended import NormalizationConfig
# from math_verify import LatexExtractionConfig, parse, verify

#format_pattern = r"^<think>(?:(?!</think>).)*</think><answer>(?:(?!</answer>).)*</answer>\Z"
format_pattern = r"^<think>(.*?)</think><answer>(.*?)</answer>"
#problem_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
#response_prefix = r"<\|im_start\|>assistant\n"

def verify_format(content):
    """
    Verify if the string meets the format requirements:
    - Must start with <think> and end with </answer>
    - Must contain exactly one pair of <think>...</think> and <answer>...</answer> tags
    - No extra characters allowed between </think> and <answer> tags
    """
    think_count = content.count("<think>")
    answer_count = content.count("<answer>")
    format_pattern = r"<think>(.*?)</think>\s*<answer>(.*?)</answer>(\s*<code>(.*?)</code>\s*<answer>(.*?)</answer>)?"
    reward_format = bool(re.match(format_pattern, content, re.DOTALL)) and think_count >= 1 and answer_count >= 1
    #print('【format reward】:',reward_format)

    return reward_format

def verify_code(content):
    pattern = r'<code>(.*?)</code>'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        reward_format=0
        pattern2 = r'"status"\s*:\s*"success"'
        # 使用 re.search 查找是否有匹配的部分
        match2 = re.search(pattern2, content)
        if match2:
            print('匹配到 "status":"success"！')
            reward_format=1
    else :
        reward_format=0
    
    return reward_format

def acc_reward(predict_str: str, ground_truth: str) -> float:
    # pattern = r"<answer>\$(.*?)\$</answer>"
    # match = re.search(pattern, predict_str)
    # if match:
    #     answer = match.group(1)  
    # else :
    #     answer= None
    pattern = r"(?:\\box\{(.*?)\})|(?:<answer>\$(.*?)\$</answer>)"
    matches = re.findall(pattern, predict_str)
    answer= None
    if matches:
        match=matches[-1]
        if match[0]:
            answer=match[0]
        elif match[1]:
            answer=match[1]
        
    #answer = extract_boxed_content(predict_str)
    if random.randint(0, 100)>98:
        try:
            print(len(predict_str),type(predict_str),":",predict_str.trip())
        except:
            pass
        print('【answer】:',answer,'|| 【ground_truth】:',ground_truth)
    reward = 1.0 if grade_answer(answer, ground_truth) else 0.0
    
    return reward


def compute_score(predict_str: str, ground_truth: str) -> float:
    acc_r = acc_reward(predict_str, ground_truth)
    format_r = 0.5 * verify_format(predict_str)
    code_r = verify_code(predict_str)
    if acc_r ==1 and format_r==0:
        format_r=0.2
    if acc_r ==1 and code_r ==1:
        acc_r = 2
    all_r= acc_r + format_r + code_r
    dict_r = {"Final_Score":all_r,"ACC":acc_r,'CODE':code_r,'FORMAT':format_r}
    json_output = json.dumps({"data": dict_r}, indent=2)
    #if code_r > 0:
    if random.randint(0, 100)>98 or code_r > 0:
        print(json_output)
    return all_r
