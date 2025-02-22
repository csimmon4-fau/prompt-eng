##
## Prompt Engineering Lab
## Platform for Education and Experimentation with Prompt NEngineering in Generative Intelligent Systems
## _pipeline.py :: Simulated GenAI Pipeline 
## 
#  
# Copyright (c) 2025 Dr. Fernando Koch, The Generative Intelligence Lab @ FAU
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# Documentation and Getting Started:
#    https://github.com/GenILab-FAU/prompt-eng
#
# Disclaimer: 
# Generative AI has been used extensively while developing this package.
# 


import requests
import json
import os
import time
import argparse
import csv
from datetime import datetime


def load_config():
    """
    Load config file looking into multiple locations
    """
    config_locations = [
        "./_config",
        "prompt-eng/_config",
        "../_config"
    ]
    
    # Find CONFIG
    config_path = None
    for location in config_locations:
        if os.path.exists(location):
            config_path = location
            break
    
    if not config_path:
        raise FileNotFoundError("Configuration file not found in any of the expected locations.")
    
    # Load CONFIG
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()


def create_payload(model, prompt, target="ollama", **kwargs):
    """
    Create the Request Payload in the format required byt the Model Server
    @NOTE: 
    Need to adjust here to support multiple target formats
    target can be only ('ollama' or 'open-webui')

    @TODO it should be able to self_discover the target Model Server
    [Issue 1](https://github.com/genilab-fau/prompt-eng/issues/1)
    """

    payload = None
    if target == "ollama":
        payload = {
            "model": model,
            "prompt": prompt, 
            "stream": False,
        }
        if kwargs:
            payload["options"] = {key: value for key, value in kwargs.items()}

    elif target == "open-webui":
        '''
        @TODO need to verify the format for 'parameters' for 'open-webui' is correct.
        [Issue 2](https://github.com/genilab-fau/prompt-eng/issues/2)
        '''
        payload = {
            "model": model,
            "messages": [ {"role" : "user", "content": prompt } ]
        }

        # @NOTE: Taking not of the syntaxes we tested before; none seems to work so far 
        #payload.update({key: value for key, value in kwargs.items()})
        #if kwargs:
        #   payload["options"] = {key: value for key, value in kwargs.items()}
        
    else:
        print(f'!!ERROR!! Unknown target: {target}')
    return payload


def model_req(payload=None):
    """
    Issue request to the Model Server
    """
        
    # CUT-SHORT Condition
    try:
        load_config()
    except:
        return -1, f"!!ERROR!! Problem loading prompt-eng/_config"

    url = os.getenv('URL_GENERATE', None)
    api_key = os.getenv('API_KEY', None)
    delta = response = None

    headers = dict()
    headers["Content-Type"] = "application/json"
    if api_key: headers["Authorization"] = f"Bearer {api_key}"

    #print(url, headers)
    print(payload)

    # Send out request to Model Provider
    try:
        start_time = time.time()
        response = requests.post(url, data=json.dumps(payload) if payload else None, headers=headers)
        delta = time.time() - start_time
    except:
        return -1, f"!!ERROR!! Request failed! You need to adjust prompt-eng/config with URL({url})"

    # Checking the response and extracting the 'response' field
    if response is None:
        return -1, f"!!ERROR!! There was no response (?)"
    elif response.status_code == 200:

        ## @NOTE: Need to adjust here to support multiple response formats
        result = ""
        delta = round(delta, 3)

        response_json = response.json()
        if 'response' in response_json: ## ollama
            result = response_json['response']
        elif 'choices' in response_json: ## open-webui
            result = response_json['choices'][0]['message']['content']
        else:
            result = response_json 
        
        return delta, result
    elif response.status_code == 401:
        return -1, f"!!ERROR!! Authentication issue. You need to adjust prompt-eng/config with API_KEY ({url})"
    else:
        return -1, f"!!ERROR!! HTTP Response={response.status_code}, {response.text}"
    return


def write_to_csv(payload, response, time_taken, prompt_type, filename="output.csv", delete_file=False):
    """
    Write the payload, response, time taken, and prompt type to a CSV file.
    """
    fieldnames = ['timestamp', 'model', 'prompt', 'prompt_type', 'temperature', 'num_ctx_tokens', 'num_output_tokens', 'time_taken', 'response']
    
    # Delete the file if delete_file is True
    if delete_file and os.path.isfile(filename):
        os.remove(filename)
    
    # Check if the file exists to write headers
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write the header only if the file does not exist
        if not file_exists:
            writer.writeheader()
        
        # Extract payload details
        model = payload.get('model', '')
        prompt = payload.get('prompt', '')
        temperature = payload.get('options', {}).get('temperature', '')
        num_ctx_tokens = payload.get('options', {}).get('num_ctx', '')
        num_output_tokens = payload.get('options', {}).get('num_predict', '')
        
        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Write the row
        writer.writerow({
            'timestamp': timestamp,
            'model': model,
            'prompt': prompt,
            'prompt_type': prompt_type,
            'temperature': temperature,
            'num_ctx_tokens': num_ctx_tokens,
            'num_output_tokens': num_output_tokens,
            'time_taken': time_taken,
            'response': response            
        })


###
### DEBUG
###
if __name__ == "__main__":
    from _pipeline import create_payload, model_req
    from prompts import get_prompt     

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Select prompt type")
    parser.add_argument(
        "prompt_type", 
        nargs="?",  # Makes this argument optional
        choices=["default", "zero_shot", "few_shot","self_reflective"], 
        default="default",  # Sets "default" if no argument is provided
        help="Type of prompt to use (default, zero_shot, few_shot, self_reflective). Defaults to 'default' if not specified."
    )
    
    args = parser.parse_args()

    # Get the prompt based on the command-line argument (default if none provided)
    prompt, num_ctx_tokens, num_output_tokens = get_prompt(args.prompt_type)

   # Create payload
    payload = create_payload(
        target="ollama",   
        model="llama3.2:latest", 
        prompt=prompt, 
        temperature=1.0, 
        num_ctx=num_ctx_tokens, 
        num_predict=num_output_tokens
    )

    # Make request and print response
    time, response = model_req(payload=payload)
    print(response)
    if time:
        print(f"Time taken: {time}s")
        
    # Write the payload, response, and time taken to a CSV file
    write_to_csv(payload, response, time, args.prompt_type,"output.csv", True)
