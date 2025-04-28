# # # import torch
# # # local_batch_size = 3
# # # rloo_k = 4

# # # rlhf_reward = torch.tensor([
# # #     1, 2, 3, # first rlhf reward for three prompts
# # #     2, 3, 4, # second rlhf reward for three prompts
# # #     5, 6, 7, # third rlhf reward for three prompts
# # #     8, 9, 10, # fourth rlhf reward for three prompts
# # # ]).float() # here we have 3 prompts which have 4 completions each

# # # rlhf_reward = rlhf_reward.reshape(rloo_k, local_batch_size)
# # # print(rlhf_reward)
# # # print(rlhf_reward.sum(0) )
# # # baseline = (rlhf_reward.sum(0) - rlhf_reward) / (rloo_k - 1)
# # # print(baseline)
# # # vec_advantages = rlhf_reward - baseline
# # # print(vec_advantages)
import json
import pandas as pd
from pprint import pprint
data_path ="/vepfs-cnbj3fa964354bf4/zyd/gemin2volczyd/verl-main/verl-main/data/math345_8k_code.parquet"  #"/gemini-1/space/xianzy/lmm-r1/examples/data/math345_8k.parquet"#
#data_path ="/gemini-1/space/zyd/verl-main/verl-main/data/torl_data/train.parquet" # "/gemini-1/space/zyd/verl-main/verl-main/data/geo3k/train.parquet" #"/gemini-1/space/zyd/Search-R1/data/nq_search/top10k_data.parquet"
df = pd.read_parquet(data_path)
num_rows = df.shape[0]
print(f"Number of rows in the dataframe: {num_rows}")
#print(df.head())
#print(df.loc[0])
asd = df.iloc[2][:].to_dict()
pprint(asd)
 
#print("prompt: ",df.loc[0,"prompt"])
# print("================")
# print(df.iloc[0][1])


# import torch

# # 创建一个示例 tensor
# tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# # 直接打印 tensor
# print(tensor)
# print(tensor.unique())
#i=1
#print(df.loc[i,'original_question'])

# # # df_top10 = df.head(10000)

# # # # 3. 保存到新文件
# # # df_top10.to_parquet("/gemini-1/space/zyd/Search-R1/data/nq_search/top10k_data.parquet", index=False)  # index=False 避免保存索引列

# from transformers import AutoModel
# model = AutoModel.from_pretrained("/gemini-1/space/zyd/Search-R1/e5-base-v2", local_files_only=True)
