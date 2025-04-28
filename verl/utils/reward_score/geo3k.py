# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from mathruler.grader import extract_boxed_content, grade_answer

ps=''' [response] <think>
Given that \(\overline{A D}\) is perpendicular to \(\overline{B C}\) and \(\overline{A B}\) is perpendicular to \(\overline{A C}\), we can conclude that \(\triangle A B C\) is a right triangle with \(\angle B A C\) as the right angle. This means that \(A D\) is the altitude from the right angle to the hypotenuse, and \(B D\) and \(C D\) are the segments of the hypotenuse. We can use the property of right triangles that the product of the segments of the hypotenuse is equal to the square of the altitude to the hypotenuse. Therefore, \(B D \times C D = A D^2\). Let's assume \(B D = x\) and \(C D = y\). Then we have \(x \times y = 10^2 = 100\). We also know that \(B D + C D = B C\), so \(x + y = B C\). To find the values of \(x\) and \(y\), we can use the fact that \(x\) and \(y\) are the roots of the quadratic equation \(t^2 - (x+y)t + xy = 0\). Let's assume \(x = 5\) and \(y = 20\), then \(5 \times 20 = 100\), and \(5 + 20 = 25\). Therefore, \(B C = 25\).
</think>
\[
\boxed{25}
\] 
My previous action is invalid. If I want to run python code, generate a Python code snippet that performs the specified operation on the provided data. Please think step by step. Ensure to break down the process into clear, logical steps. Make sure to print the final result in the generated code snippet with a descriptive message explaining what the output represents. The final output should be presented in the following format:
<code>
```python
<some code snippet here>
```
</code>
If I want to give the final answer, I should put the answer between <answer> and </answer>

<code_result>{'status': 'error', 'output': 'Restricted code compilation failed: ("Line 1: SyntaxError: invalid syntax at statement: \'<some code snippet here>\'",)'}</code_result>'''


def format_reward(predict_str: str) -> float:
    #import pdb;pdb.set_trace()
    pattern = re.compile(r'<think>.*</think>.*\\boxed\{.*\}.*|'
                         r'<code>```python.*```</code>.*\\boxed\{.*\}.*', re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0

def acc_reward(predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict_str)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(predict_str: str, ground_truth: str) -> float:
    return 0.9 * acc_reward(predict_str, ground_truth) + 0.1 * format_reward(predict_str)
