import pickle
import os.path as osp
import os
import torch
from transformers import BertTokenizer
from lavis.models import load_model_and_preprocess
from tqdm import tqdm

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="./cache")

path = ""
en_fr = "en(fr)_dbp15k_link_img_dict_full.pkl"
path_ent_ids = "/home/2022licx/data/mmkg/DBP15K/fr_en/ent_ids_2"

# Load entity IDs mapping with progress bar
entity_ids = {}
with open(path_ent_ids, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    total_lines = len(lines)
    for line in tqdm(lines, desc="Loading entity IDs", total=total_lines, mininterval=0.5):
        parts = line.strip().split('\t')
        if len(parts) == 2:
            entity_id, entity_url = parts
            entity_ids[entity_url] = entity_id

print("finish ent_ids")

# Load the full dictionary of entities and their images
enfr_images = pickle.load(open(osp.join(path, en_fr), 'rb'))

print("finish pkl img_dict")

# Define target entity URLs from the loaded dictionary (e.g., first two keys)
target_entity_urls = list(enfr_images.keys())

print("finish target_entity_url")

# Prepare to store results
results = {}

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BLIP large captioning model finetuned on COCO directly to GPU
model, vis_processors, _ = load_model_and_preprocess(
    name="blip_caption", model_type="large_coco", is_eval=True, device=device
)

print("Model loaded on device:", device)

# Process each entity with progress bar
for i, target_entity_url in enumerate(tqdm(target_entity_urls, desc="Processing entities")):
    entity_id = entity_ids.get(target_entity_url)

    if entity_id:
        try:
            # Load associated image
            raw_image = enfr_images[target_entity_url].convert("RGB")

            # Prepare the image as model input using the associated processors
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

            # Generate multiple captions using nucleus sampling
            captions = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=3)

            # Store the results
            results[entity_id] = captions

        except Exception as e:
            print(f"Error processing {target_entity_url}: {e}")

print("finish generat entity captions")

# Prepare the output string
output_str = ""
for entity_id, captions in results.items():
    output_str += f"{entity_id}: {captions}\n"

# Define output file path
output_file = ""

# Write output to file
with open(output_file, 'w', encoding='utf-8') as file:
    file.write(output_str)

print(f"Output saved to {output_file}")

