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
# The above copyright notice and this permission shall be included in all
# copies or substantial portions of the Software.
# 
# Documentation and Getting Started:
#    https://github.com/GenILab-FAU/prompt-eng
#
# Disclaimer: 
# Generative AI has been used extensively while developing this package.
# 


# Standard library imports
import os
import json
import time
import argparse
import csv
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# Third-party imports
import requests
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
from prompts import self_reflective_prompt, get_prompt

# Constants
FILENAME = "output.csv"  # Output file for logging responses
DELETE_FILE = True      # If True, deletes existing file before writing
FIELDNAMES = [          # CSV column headers
    'timestamp',        # Time of request
    'model',           # Model name used
    'prompt',          # Input prompt
    'prompt_type',     # Type of prompt (default, zero-shot, etc.)
    'temperature',     # Model temperature setting
    'num_ctx_tokens',  # Context window size
    'num_output_tokens', # Maximum tokens to generate
    'time_taken',      # Request processing time
    'similarity',      # Semantic similarity score
    'clarity',         # Clarity score
    'specificity',     # Specificity score
    'effectiveness',   # Effectiveness score
    'response'         # Model response (moved to last)
]
_FIRST_WRITE = True  # Track first write operation

# Model initialization
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

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
    # print(payload)
    print(f"\nPayload: {payload}\n")

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

from prompts import self_reflective_prompt

def generate_self_reflective_prompt(initial_prompt, model, max_iterations, temperature, num_ctx_tokens, num_output_tokens, similarity_threshold=0.95):
    """
    Generate a self-reflective prompt by iteratively calling the model and refining the prompt.
    """
    prompt = initial_prompt 
    best_prompt = initial_prompt
    best_similarity = 0

    for i in range(max_iterations):
        print(f"Iteration {i+1}: Refining prompt...")
        
        # Create a self-reflective version of the current prompt
        reflective_prompt = self_reflective_prompt(prompt)  # This creates the meta-prompt
        
        # Process the self-reflective prompt using process_request
        payload, response, time = process_request(
            prompt=reflective_prompt,  # Use the reflective version
            model=model,
            temperature=temperature,
            num_ctx_tokens=num_ctx_tokens,
            num_output_tokens=num_output_tokens
        )
        
        refined_prompt = refine_prompt(prompt, response)
        similarity = compute_similarity(prompt, refined_prompt)
        
        # Calculate metrics for logging
        clarity, specificity, effectiveness = evaluate_prompt(prompt)
        
        # Log results with metrics
        log_results(
            payload=payload,
            response=response,
            time_taken=time,
            prompt_type=f"self_reflective_iteration_{i+1}",
            similarity=similarity,
            clarity=clarity,
            specificity=specificity,
            effectiveness=effectiveness
        )
        
        if similarity >= similarity_threshold:
            print(f"Convergence reached at iteration {i+1}")
            best_prompt = refined_prompt
            break
            
        if similarity > best_similarity:
            best_similarity = similarity
            best_prompt = refined_prompt
            
        prompt = refined_prompt  # Use refined prompt for next iteration
    
    return best_prompt


def refine_prompt(prompt, response):
    """
    Refine the prompt based on the response.
    This function assumes that the response contains only the refined prompt.
    """
    # Debug: Print the response to verify its content
    # print(f"XXXResponse: {response}")

    # Assume the entire response is the refined prompt
    refined_prompt = response.strip()
    
    return refined_prompt


def process_request(prompt, model, temperature, num_ctx_tokens, num_output_tokens, target="ollama"):
    """
    Process the request by creating the payload and making the model request.
    """
    # Create payload
    payload = create_payload(
        target=target,   
        model=model, 
        prompt=prompt, 
        temperature=temperature, 
        num_ctx=num_ctx_tokens, 
        num_predict=num_output_tokens    
    )

    # Make request and get response
    time, response = model_req(payload=payload)
    print(f"\nModel Response: {response}")    
    print(f"\nTime taken: {time}s")
    
    return payload, response, time

def log_results(payload: Dict[str, Any], response: str, time_taken: float, prompt_type: str, 
               similarity: float = 0.0, clarity: int = 0, specificity: int = 0, 
               effectiveness: int = 0) -> None:
    """
    Log the payload, response, time taken, prompt type and evaluation metrics to a CSV file.
    """
    global FILENAME, DELETE_FILE, _FIRST_WRITE
    
    print(f"\nLog operation - DELETE_FILE: {DELETE_FILE}, _FIRST_WRITE: {_FIRST_WRITE}")
    
    if DELETE_FILE and _FIRST_WRITE and os.path.isfile(FILENAME):
        print(f"\nDeleting existing output file: {FILENAME}")
        os.remove(FILENAME)
    _FIRST_WRITE = False  # Move this outside the if block
    
    file_exists = os.path.isfile(FILENAME)
    
    print(f"\nWrite to log")
    with open(FILENAME, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)  # Use constant
        
        if not file_exists:
            writer.writeheader()
        
        # Extract payload details
        model = payload.get('model', '')
        prompt = payload.get('prompt', '')
        temperature = payload.get('options', {}).get('temperature', '')
        num_ctx_tokens = payload.get('options', {}).get('num_ctx', '')
        num_output_tokens = payload.get('options', {}).get('num_predict', '')
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        writer.writerow({
            'timestamp': timestamp,
            'model': model,
            'prompt': prompt,
            'prompt_type': prompt_type,
            'temperature': temperature,
            'num_ctx_tokens': num_ctx_tokens,
            'num_output_tokens': num_output_tokens,
            'time_taken': time_taken,
            'similarity': similarity,
            'clarity': clarity,
            'specificity': specificity,
            'effectiveness': effectiveness,
            'response': response
        })

def process_and_log_request(prompt, model, prompt_type, temperature, num_ctx_tokens, num_output_tokens):
    """Process the request and log the response with metrics to a CSV file."""
    # Process the request
    payload, response, time = process_request(
        prompt=prompt,
        model=model,
        temperature=temperature,
        num_ctx_tokens=num_ctx_tokens,
        num_output_tokens=num_output_tokens
    )
    
    # Calculate metrics
    clarity, specificity, effectiveness = evaluate_prompt(prompt)
    similarity = compute_similarity(prompt, response) if response else 0.0
    
    # Log the results with metrics
    log_results(
        payload=payload,
        response=response,
        time_taken=time,
        prompt_type=prompt_type,
        similarity=similarity,
        clarity=clarity,
        specificity=specificity,
        effectiveness=effectiveness
    )
    
    return response, time

def get_embedding(text):
    """
    Get BERT embeddings for text, truncating if necessary.
    
    Args:
        text: Input text to embed
        
    Returns:
        numpy.ndarray: Text embedding
    """
    inputs = bert_tokenizer(
        text, 
        return_tensors='pt',
        truncation=True,
        max_length=512  # BERT's maximum sequence length
    )
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def compute_similarity(prompt1, prompt2):
    embedding1 = get_embedding(prompt1)
    embedding2 = get_embedding(prompt2)
    return cosine_similarity(embedding1, embedding2)[0][0]

def evaluate_prompt(prompt: str) -> Tuple[int, int, int]:
    """
    Evaluate the quality metrics of a prompt.
    
    Args:
        prompt: The prompt text to evaluate
        
    Returns:
        Tuple[int, int, int]: (clarity_score, specificity_score, effectiveness_score)
    """
    try:
        clarity_score = len(prompt.split())
        specificity_score = prompt.count('specific')
        effectiveness_score = prompt.count('effective')
        
        return clarity_score, specificity_score, effectiveness_score
    except Exception as e:
        print(f"Error evaluating prompt: {e}")
        return 0, 0, 0

#------------------------------------------------------------------------------
# DEBUG SECTION
# This section runs when the script is executed directly
#------------------------------------------------------------------------------
if __name__ == "__main__":
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
    prompt,model,temperature,num_ctx_tokens, num_output_tokens = get_prompt(args.prompt_type)

    #generate the self reflective prompt
    if args.prompt_type == "self_reflective":
        # Generate self-reflective prompt
            prompt = generate_self_reflective_prompt(
            initial_prompt=prompt,
            model=model,
            max_iterations=5,  
            temperature=temperature,
            num_ctx_tokens=num_ctx_tokens,
            num_output_tokens=num_output_tokens
        )
        
      
    # Send the prompt to the model
    print(f"\nFinal Prompt Executing")
    
    #Process the request
    payload, response, time = process_request(
        prompt=prompt,
        model=model,
        temperature=temperature,
        num_ctx_tokens=num_ctx_tokens,
        num_output_tokens=num_output_tokens
    )
    
    # Log results (with default metrics)
    log_results(
        payload=payload,
        response=response,
        time_taken=time,
        prompt_type=args.prompt_type
    )


