from openai import OpenAI
import json
import time
import random
from tqdm import tqdm
import re

API_KEY = ""
BASE_URL = ""

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def generate_prompt(desc1, desc2):
    return f"""Here are two descriptions of different entities:

Description 1:
"{desc1}"

Description 2:
"{desc2}"

Do they describe the same topic? Answer only "Yes" or "No". 
If Yes, summarize the shared theme in one sentence.
If No, leave the theme blank."""

def call_gpt_with_retry(prompt, retries=3, delay=3):
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"[WARN] GPT call failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay)
    return None


def parse_response(response):
    if not response:
        return 0, ""

    response = response.strip()

    # 判断是否以 Yes 开头
    if re.match(r'(?i)^yes[\.,:]*', response):
        # 提取 Yes 后面那一整句作为主题
        match = re.match(r'(?i)^yes[\.,:]*\s*(.*)', response)
        context = match.group(1).strip() if match else ""
        return 1, context

    return 0, ""

def process(input_json, output_json):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(len(data))
    results = []
    for item in tqdm(data, desc="Processing pairs"):
        id1 = item['id1']
        id2 = item['id2']
        desc1_list = item.get('desc1', [])
        desc2_list = item.get('desc2', [])

        if desc1_list and desc2_list:
            desc1 = random.choice(desc1_list)
            desc2 = random.choice(desc2_list)
            prompt = generate_prompt(desc1, desc2)
            response = call_gpt_with_retry(prompt)
            mask, context = parse_response(response)
            time.sleep(1.2)
        else:
            mask = 0
            context = ""

        results.append({
            "id1": id1,
            "id2": id2,
            "mask": mask,
            "context": context
        })

    with open(output_json, 'w', encoding='utf-8') as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)
    print(f"Saved to {output_json}")


process('aligned_output.json', 'output_with_mask_context.json')
