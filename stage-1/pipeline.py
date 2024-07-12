from datasets import load_dataset
import json
import requests
import base64
from PIL import Image
from io import BytesIO
import time
import os
import sys
import shutil
import random
import re
from call import *

prompt = {
    'what': 'What is in this picture?',
    'describe': 'Describe the picture',
    'details': 'Please rewrite the sentence describing the image in more detail, clarity and short in English only so that the generated image is highly detailed and sharp focus. Description: ',
    'happy': 'Describe this picture happier, with smile and laugh',
    'list': 'List the things that appeared in this picture, also list the relevant objects',
    'verify': 'Please let me know whether the image is of good quality and suitable with the description or not. Answer with only a \"Yes\" or \"No\". Please output in json format: {{"answer": <your answer>}}. No explanation needed. Description: ',
    'extend': 'Stable Diffusion is an AI art generation model similar to DALLE-2. Please write me a detailed prompt for generating art with Stable Diffusion exactly about the description follow the following rules:\
        - Prompt should always be written in English, regardless of the input language. Please provide the prompt in English.\
        - Prompt should consist of a description of the scene followed by modifiers divided by commas.\
        - When generating description, focus on portraying the visual elements rather than delving into abstract psychological and emotional aspects. Provide clear and concise details that vividly depict the scene and its composition, capturing the tangible elements that make up the setting.\
        - The modifiers should alter the mood, style, lighting, and other aspects of the scene.\
        - Multiple modifiers can be used to provide more specific details.\
        Please write only exactly a prompt.\
        Description: ',
    
    'limited_details': 'Please improve the description image more highly detailed and shortly in English. Description: ',
    'verify_2': 'Please let me know if the image is good quality and suitable with the description or not, answer Yes if so.',
    'feedback': 'If no, what things the description has but the image doesn\'t have, answer shortly. Description: '
}

SD_URL = 'http://127.0.0.1:8000/api'
LLAVA_URL = 'http://localhost:11434/api/generate'
ORIGINAL_FOLDER = 'sample_appendix/ground_truth'
OUTPUT_FOLDER = 'sample_appendix/gen_image'
JSON_FOLDER = 'sample_appendix/'

def saveBase64(base64str, path="sample.png"):
    decoded_data = base64.b64decode(base64str)
    image_stream = BytesIO(decoded_data)
    image = Image.open(image_stream)
    image.save(path)
    print(f"\t-> Image saved as {path} {image.size}")

def readImageFromUrl(url):
    response = requests.get(url)
    if response.status_code == 200:
        image_data = response.content
        base64_encoded = base64.b64encode(image_data).decode('utf-8')
        return base64_encoded
    else:
        print(f"Failed to fetch the image. Status code: {response.status_code}")
        return None

def accept(response):
    # Normalize the response to lower case for comparison
    response_normalized = response.strip().lower()
    
    # Check if the response contains the exact word "yes"
    if response_normalized == "yes":
        return True
    elif response_normalized.startswith('{"answer":'):
        try:
            response_dict = json.loads(response)
            return response_dict.get("answer", "").strip().lower() == "yes"
        except json.JSONDecodeError:
            return False
    else:
        # Check for the case where response is a JSON string wrapped in quotes
        match = re.search(r'{"answer":\s*"(yes|no)"}', response_normalized)
        if match:
            return match.group(1) == "yes"
    
    return False

def create(image, input, target=prompt['details'], max_iter=10, id='sample'):
    id = str(id)
    cost = []
    generating_process = []
    i = 1
    input0 = input
    
    llava_respone = ''

    savePath = os.path.join(ORIGINAL_FOLDER, f"{id}.png")
    saveBase64(image, path=savePath)
    
    # Set verifying prompt to ask llava if the generated image suits with input
    # verifying_prompt = prompt['verify_2'] + prompt['feedback'] + '\"' + input + '\"'
    verifying_prompt = prompt['verify'] + '\"' + input + '\"'
    while i <= max_iter:

        print(f"\nIter: {i}")
        
        # Improve input_0
        sd_prompt, process_time = getResponeFromLLaVA13b(target + '\"' + input + '\"')
        cost.append(process_time)
        # Generate image for iteration i
        print(f"\t[Generating image] ({sd_prompt[:50]}...) ...")
        image, process_time = getImageFromSD(sd_prompt)
        cost.append(process_time)
        print(f"\t[Generating image][DONE] in {process_time:0.1f} secs:")
        
        # Verify generated image by llava
        llava_respone, process_time = getResponeFromLLaVA13b(verifying_prompt, image)
        cost.append(process_time)

        generating_process.append((i, llava_respone))
        
        if accept(llava_respone):
            break
        i+=1

    # Save image 
    savePath = os.path.join(OUTPUT_FOLDER, f"{id}.png")
    saveBase64(image, path=savePath)

    ret = {
        'iterations': i,
        'input': input,
        'improve': sd_prompt,
        'cost': sum(cost),
        'generating_process': generating_process
    }
    
    # Save json file
    with open(os.path.join(JSON_FOLDER, f'{id}.json'), 'w') as jsonfile:
        json.dump(ret, jsonfile, indent=4)
    return ret


if __name__ == '__main__':
    start_time = time.time()
    with open("map_result.json", 'r') as f:
        dataset = json.load(f)

    costs = []
    iters = []
    CHECKPOINT = 0

    # For resuming, please type the previous values to argv
    # python3 pipeline.py CHECKPOINT PREV_TOTAL_COST PREV_AVG_COST PREV_AVG_ITER
    if len(sys.argv) > 4:
        CHECKPOINT = int(sys.argv[1])
        prev = {
            'totalCost': float(sys.argv[2]),
            'avgCost': float(sys.argv[3]),
            'avgIter': float(sys.argv[4]),
        }
        print(f"\t\tResume from {CHECKPOINT} - Loading previous log values:")
        print(prev)
        costs = [0 for i in range(CHECKPOINT)]
        iters = [0 for i in range(CHECKPOINT)]
        costs.append(prev['totalCost'])
        iters.append(prev['avgIter'] * CHECKPOINT)

    i = 1
    for image_id in dataset:
        if i < CHECKPOINT:
            i+=1
            continue
        image_url = dataset[image_id]['image_url']
        image0 = readImageFromUrl(image_url)
        input0 = random.choice(dataset[image_id]['prompts'])

        print(f'\t•[{i}] URL: ', image_url)
        print(f'\t•[{i}] Description: ', input0)
    
        result = create(image0, input0, target=prompt['details'], max_iter=5, id=image_id)
        
        print('\n')
        costs.append(result['cost'])
        iters.append(result['iterations'])

        print(f"\t• Total cost: {sum(costs):0.1f} secs")
        print(f"\t• Average cost: {sum(costs)/len(costs):0.1f} secs")
        print(f"\t• Average iterations: {sum(iters)/len(iters):0.1f} iters")
        i+=1
        # if i>10: break

    
    if len(sys.argv) > 4:
        print(f"True cost: {int(sys.argv[2]) + time.time() - start_time}")
    else:
        print(f"True cost: {time.time() - start_time}")

