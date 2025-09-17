from llm2vec import LLM2Vec
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
import pickle
import numpy as np


tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
config = AutoConfig.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)
model = PeftModel.from_pretrained(model,"McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",)
model = model.merge_and_unload()
model = PeftModel.from_pretrained(model, "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised")
l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)

prompt = """
You are an expert who can provide concise explanations based on entity information. 
I will give you the properties of attributes and values of an entity in the form of (predicate: object). 
Using this information along with your general knowledge, please provide a short description of the entity.

- The explanation should be no longer than 100 words.
- Focus on summarizing the entity based on the given information and your general knowledge.
- Do not include unnecessary details or explanations beyond the entity description.
- Do not include entity name.

Example:

Entity Information: Albert Einstein (profession: Physicist), (known for: Theory of Relativity)
Explanation: **This entity** was a renowned physicist best known for developing the Theory of Relativity, a fundamental theory in modern physics.

Now, please summarize the following entity information and return an desctription in English:
"""


instruction = (
    prompt
)

name_values_left = {}
name_values_right = {}

entids = {}
with open('ent2id', 'r', encoding='utf-8') as e2i:
    lines = e2i.readlines()

for line in lines:
    id, name = line.strip().split('\t')
    entids[id] = name

with open('att1.txt', 'r', encoding='utf-8') as txf:
    lines = txf.readlines()

    for i, line in enumerate(tqdm(lines, desc="Processing left entities")):
        id, info = line.strip().split('\t')
        info = eval(info)
        desc = ";".join(info)
        desc += entids[id] +' ' + desc
        name_values_left[id] = []
        name_values_left[id].append(instruction)
        name_values_left[id].append(desc)

with open('att2.txt', 'r', encoding='utf-8') as txf:
    lines = txf.readlines()
        
    for i, line in enumerate(tqdm(lines, desc="Processing right entities")):
        id, info = line.strip().split('\t')
        info = eval(info)
        desc = ";".join(info)
        desc += entids[id] +' ' + desc
        name_values_right[id] = []
        name_values_right[id].append(instruction)
        name_values_right[id].append(desc)

keys_left = list(name_values_left.keys())
embed_left = l2v.encode(list(name_values_left.values()))
value_left = np.array(embed_left)

keys_right = list(name_values_right.keys())
embed_right = l2v.encode(list(name_values_right.values()))
value_right = np.array(embed_right)

for i, key in enumerate(keys_left):
    name_values_left[key] = value_left[i]
for i, key in enumerate(keys_right):
    name_values_right[key] = value_right[i]

with open('att_sum_left.pkl', 'wb') as f:
    pickle.dump(name_values_left, f)
with open('att_sum_right.pkl', 'wb') as f:
    pickle.dump(name_values_right, f)

embed_left = torch.nn.functional.normalize(torch.tensor(embed_left).clone().detach(), p=2, dim=1)
embed_right = torch.nn.functional.normalize(torch.tensor(embed_right).clone().detach(), p=2, dim=1)

cos_sim = torch.mm(embed_left, embed_right.transpose(0, 1)).numpy()


def calculate_metrics_with_mapping(sim_matrix, left_keys, right_keys, ground_truth, k_list=[1, 5, 10]):
    hits = {k: 0 for k in k_list}
    mrr = 0
    num_queries = sim_matrix.shape[0]

    ground_truth_dict = {left: right for left, right in ground_truth}

    for i in range(num_queries):
        left_id = left_keys[i]
        if left_id not in ground_truth_dict:
            continue
        true_right_id = ground_truth_dict[left_id]

        ranks = np.argsort(-sim_matrix[i])
        sorted_right_ids = [right_keys[j] for j in ranks]

        for k in k_list:
            if true_right_id in sorted_right_ids[:k]:
                hits[k] += 1

        if true_right_id in sorted_right_ids:
            rank_index = sorted_right_ids.index(true_right_id)
            mrr += 1 / (rank_index + 1)

    for k in k_list:
        hits[k] /= num_queries
    mrr /= num_queries

    return hits, mrr


ground_truth = []
with open('ill_ent_ids', 'r', encoding='utf-8') as f:
    for line in f:
        left_id, right_id = line.strip().split('\t')
        ground_truth.append((left_id, right_id))

hits, mrr = calculate_metrics_with_mapping(cos_sim, keys_left, keys_right, ground_truth)
print(f"Hits@1: {hits[1]}\nHits@5: {hits[5]}\nHits@10: {hits[10]}\nMRR: {mrr}")
