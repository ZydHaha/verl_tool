import torch
import re,json
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
# from search_r1.utils import set_seed
# from search_r1.utils.plot import (
#     save_trajectory_to_output,
#     parse_llm_output
# )
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
import pprint

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    # logging: dict
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        # logger: Tracking,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        # self.logger = logger
        self.is_validation = is_validation
    
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        #将从环境中获取的观察结果字符串列表转换为适合模型输入的张量格式
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses = self.tensor_fn.concatenate_with_padding([
                right_side['responses'],
                cur_responses,
                next_obs_ids
            ], pad_to_left=False)
        else:
            responses = self.tensor_fn.concatenate_with_padding([
                right_side['responses'],
                cur_responses,
            ], pad_to_left=False)
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        #print("【active_batch size:】", active_batch.batch['input_ids'].shape[0])
        for key in active_batch.batch.keys():  #
             active_batch.batch[key] = active_batch.batch[key].long()#
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        # Print debug info
        #print("Number of GPUs:", num_gpus)
        #print("Input batch size:", batch_size)
        #print("Remainder:", remainder)
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
       # print("【padded_active_batch size:】", padded_active_batch.batch['input_ids'].shape[0])
        for key in padded_active_batch.batch.keys():#
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()#
###  Actor 模型的工作组 生成序列
        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        # try:
        #     print("【trimmed_batch size:】", len(trimmed_batch))
        # except:
        #     pass
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)#标记当前批次中哪些样本仍然处于“活跃”状态（即未完成生成）
        active_num_list = [active_mask.sum().item()]#记录每一步活跃样本的数量
        #print("active_num_list:",active_num_list)
        rollings = gen_batch
#多轮交互，直到达到上限或所有样本都完成生成
        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )#裁剪到有效长度 填充（padding）部分
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items() #返回字典中所有的键值对
            })            
            #print("Before _generate_with_gpu_padding:")
            #print("Batch size:", rollings_active.batch['input_ids'].shape[0])

            gen_output = self._generate_with_gpu_padding(rollings_active) #生成新的序列
            #print("After _generate_with_gpu_padding:")
            #print("Output batch size:", gen_output.batch['responses'].shape[0])#gen_output:  torch.Size([5120, 2048])
            
            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            #print(responses_ids.shape,active_mask.sum())#torch.Size([5120, 2054]) tensor(1024) responses.shape[0] == sum
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
             #对生成结果进行填充（padding），确保批次内的样本长度一致

            # 执行预测并与环境交互,responses_str有《search》的，进行search ，并且返回《information》
            next_obs, dones = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )
            print("【responses_str】: ",responses_str[1].strip(),"\n【next_obs】: ",next_obs[1])
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())

            next_obs_ids = self._process_next_obs(next_obs) #token化
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )#更新搜索结果后继续下一轮推理
            
        # final LLM rollout
        if active_mask.sum(): #如果仍有活跃样本，则进行最后一次推理，逻辑与主循环类似。
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )

        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # 拼接后的 attention_mask 可以用于轨迹采样和奖励计算
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
            #create_attention_mask 的逻辑会检查每个 token 是否为填充符（如 pad_token_id），并生成相应的掩码
        ], dim=1)  #[batch_size, sequence_length]
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output
    
#execute_predictions  根据预测的动作，在多个环境中执行相应的操作，并生成观察结果和完成状态。它支持两种主要动作：
    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=False,run_code=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions) #'search' 或 'answer' 以及对应的内容
        
        #print("cur_actions",cur_actions)
        next_obs, dones = [], []

        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        code_to_run = [content for action, content in zip(cur_actions, contents) if action == 'code']
#！！！进行搜索
        if do_search and search_queries :
            search_results = self.batch_search(search_queries) 
            print(f"【search_queries】: {search_queries}\nsearch_results: {search_results}")
            #import pdb;pdb.set_trace()
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])
#！！！执行代码    
        if run_code and code_to_run :
            code_results = self.batch_code_run(code_to_run) 
            try:
                #import pdb;pdb.set_trace()
                datatemp = []
                for i in range(5):
                    datatemp.append({
                        "index": i,
                        "code_to_run": code_to_run[i],
                        "code_results": code_results[i]
                    })
                json_output = json.dumps({"【codes": datatemp}, indent=2)
                print(json_output)
                #print(f"【code_to_run】: {code_to_run[:5]}\n【code_results】: {code_results[:5]}")
            except:
                pass
            #import pdb;pdb.set_trace()
            assert len(code_results) == sum([1 for action in cur_actions if action == 'code'])
        else:
            code_results = [''] * sum([1 for action in cur_actions if action == 'code'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
#！！！！添加搜索结果到 next_obs, 类似续写方法中添加提示！！！！
            if not active:
                next_obs.append('')
                dones.append(1)
            else:
                if action == 'answer':
                    import random ####增加写代码概率
                    if random.randint(0, 10)>7:
                        next_obs.append(f'''\nWait! If I want to generate code snippets and execute the code to make exact calculations, write the complete code directly between the <code> and </code> tags. Please note:
1. The code must use print to output the final answer.
2. After writing, the code will be executed, and you need to wait for the execution result to get the answer.\n''')                      
                        #next_obs.append(f'''\nWait! If I want to call a tool to generate code snippets and execute code for percise calculation, \
                          #  I don't need to write code but only to give the calculation task description between <code> and </code>. \n''')
                        dones.append(0)
                    else:
                        next_obs.append('')
                        dones.append(1)
                elif do_search and action == 'search':
                    # print(i,predictions[i])
                    # import pdb;pdb.set_trace()
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    #print(next_obs[:-1])
                    dones.append(0)
                elif action == 'code':
                    #print("i,predictions[i]",i,predictions[i])
                    # import pdb;pdb.set_trace()
                    next_obs.append(f'\n\n<code_result>{code_results.pop(0)}</code_result>\n\n')
                    #print(next_obs[:-1])
                    dones.append(0)
                else: #If I want to search, I should put the query between <search> and </search>. \
#                     next_obs.append(f'''\nMy previous action is invalid. \
# If I want to give the final answer, I should put the answer between <answer> and </answer>
# If I want to call a code agent to generate code snippets and execute code for percise calculation, I don't need to write code but only give the calculation task description between <code> and </code>. \
#     Let's try again\n''')
                    next_obs.append(f'''\nMy previous action is invalid. \
If I want to give the final answer, I should put the answer between <answer> and </answer>
If I want to run python code to verify the calculations, make sure to use python fucntion "print()" to print the final result in the generated code with a descriptive message explaining what the output represents.\
The final output should be presented in the following format: <code>```python<code here>```</code>.\n
<code>```python''')
                    dones.append(0)
            
        assert len(code_results) == 0
            
        return next_obs, dones

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []

        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer|code)>(.*?)</\1>'
                matches = re.findall(pattern, prediction, re.DOTALL)
                #print(matches)
                if matches:
                    #print(matches)
                    match=matches[-1]  #找到最后一个关键字
                    if len(matches)>1:
                        if matches[-1][0] == "answer" and  matches[-2][0] == "code":
                            match=matches[-2] #找到code组
                        
                    #print(match[1])
                    #match = re.search(pattern, prediction, re.DOTALL)
                    #print(match.group(0))
                    if match:
                        content = match[1].strip() #找到code组里的内容
                        action = match[0]
                        if content == "and" or content == " and ":
                            action = None
                        
                        # content = match.group(2).strip()  # Return only the content inside the tags
                        # action = match.group(1)
                        if action =='code':
                            match = re.search(r"```python\s*(.*?)\s*```", content, re.DOTALL)
                            if match:
                                content = match.group(1).strip()
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")

            actions.append(action)
            contents.append(content)
        # actions = []
        # contents = []
                
        # for prediction in predictions:
        #     if isinstance(prediction, str): # for llm output
        #         pattern = r'<(search|answer|code)>(.*?)</\1>'
        #         match = re.search(pattern, prediction, re.DOTALL)
        #         if match:
        #             content = match.group(2).strip()  # Return only the content inside the tags
        #             action = match.group(1)
        #             if action =='code':
        #                 match = re.search(r"```python\s*(.*?)\s*```", content, re.DOTALL)
        #                 if match:
        #                     content = match.group(1).strip()
        #         else:
        #             content = ''
        #             action = None
        #     else:
        #         raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
        #     actions.append(action)
        #     contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results] #返回字符串列表

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            #enumerate() 自动为我们生成并管理索引，省去了手动维护索引的麻烦
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference

    def batch_code_run(self, codes: List[str] = None) -> str:
        """
        Args:
            codes: codes to run
        Returns:
            codes results which is concatenated into a string
        """
        try:
            results = self._execute_batch_code(codes)
            code_results = results["results"]
        except TypeError as e:
            print(f"Execution failed: {e}")
        i=0
        for item in code_results:
            if item["status"] =='success':
                i+=1
        print(f"-------==================------------------\n共{len(code_results)}个,其中{i}个执行成功，成功率{i/len(code_results):.2f}%\n")
        return results["results"]#字典列表
    
    def _execute_batch_code(self, code_list):
        """
        测试 /execute-batch/ 接口。
        :param code_list: 用户提交的代码片段列表
        :return: 响应结果
        """
        try:
            # 构造请求体
            payload = {"code_list": code_list}
            BASE_URL = "http://127.0.0.1:1142/execute-batch/"
            #BASE_URL = "http://172.31.16.2:8000/execute-batch/"
            # 发送 POST 请求
            print("!!!!!!code_list", len(code_list))
            #import pdb;pdb.set_trace()
            response = requests.post(BASE_URL, json=payload)

            # 检查响应状态码
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: {response.status_code}, {response.text}")
                return None

        except Exception as e:
            print(f"Request failed: {e}")
            return None