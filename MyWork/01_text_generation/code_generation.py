# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    code_generation.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: agaroux <agaroux@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/02/17 15:12:44 by agaroux           #+#    #+#              #
#    Updated: 2026/02/17 16:15:09 by agaroux          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import json
import boto3
import botocore
from IPython.display import display, Markdown
import time

with open('../secrets/aws_access_key_id.txt', 'r') as f:
    KEY_ID = f.read().strip()

with open('../secrets/aws_secret_access_key.txt', 'r') as f:
    ACCESS_KEY = f.read().strip()

session = boto3.session.Session()
region = session.region_name or "eu-west-3"
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=region,
    aws_access_key_id= KEY_ID,
    aws_secret_access_key= ACCESS_KEY
)

MODEL_ID = "mistral.mistral-large-2402-v1:0"

code_generation_prompt = """
Create a Python function called get_weather that accepts a location as parameter. \
The function should return a dictionary containing weather data (condition, temperature, and humidity) for predefined cities.\
Use a mock data structure instead of actual API calls. Include New York, San Francisco, Miami, and Seattle as default cities.\
The return statement should look like the following: return weather_data.get(location, {"condition": "Unknown", "temperature": 0, "humidity": 0}).
Only return the function and no preamble or examples.
"""

converse_request = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "text": f"{code_generation_prompt}"
                }
            ]
        }
    ],
    "inferenceConfig": {
        "temperature": 0.0,
        "topP": 0.9,
        "maxTokens": 500
    }
}

def display_response(response, model_name=None):
    if model_name:
        print(f"\n### Response from {model_name}\n")
    print(response)
    print("\n" + "-"*80 + "\n")

try:
    response = bedrock.converse(
        modelId=MODEL_ID,
        messages=converse_request["messages"],
        inferenceConfig=converse_request["inferenceConfig"]
    )
    
    # Extract the model's response
    claude_converse_response = response["output"]["message"]["content"][0]["text"]
    display_response(claude_converse_response, "Claude 3.7 Sonnet (Converse API)")
except botocore.exceptions.ClientError as error:
    if error.response['Error']['Code'] == 'AccessDeniedException':
        print(f"\x1b[41m{error.response['Error']['Code']}: {error.response['Error']['Message']}\x1b[0m")
        print("Please ensure you have the necessary permissions for Amazon Bedrock.")
    else:
        raise error
    
# def get_weather(location):
#     weather_data = {
#         "New York": {"condition": "Cloudy", "temperature": 15, "humidity": 60},
#         "San Francisco": {"condition": "Foggy", "temperature": 18, "humidity": 75},
#         "Miami": {"condition": "Sunny", "temperature": 28, "humidity": 65},
#         "Seattle": {"condition": "Rainy", "temperature": 12, "humidity": 80},
#     }
#     return weather_data.get(location, {"condition": "Unknown", "temperature": 0, "humidity": 0})