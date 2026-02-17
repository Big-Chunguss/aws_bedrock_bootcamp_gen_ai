# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    weather.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: agaroux <agaroux@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/02/17 15:12:44 by agaroux           #+#    #+#              #
#    Updated: 2026/02/17 16:14:34 by agaroux          ###   ########.fr        #
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

def get_weather(location):
    weather_data = {
        "New York": {"condition": "Cloudy", "temperature": 15, "humidity": 60},
        "San Francisco": {"condition": "Foggy", "temperature": 18, "humidity": 75},
        "Miami": {"condition": "Sunny", "temperature": 28, "humidity": 65},
        "Seattle": {"condition": "Rainy", "temperature": 12, "humidity": 80},
    }
    return weather_data.get(location, {"condition": "Unknown", "temperature": 0, "humidity": 0})

function_request = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "text": "What's the weather like in San Francisco right now? And what should I wear?"
                }
            ]
        }
    ],
    "inferenceConfig": {
        "temperature": 0.0,
        "maxTokens": 500
    }
}

weather_tool = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_weather",
                "description": "Get current weather for a specific location",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name to get weather for"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        }
    ],
    "toolChoice": {
        "auto": {}
    }
}

response = bedrock.converse(
    modelId=MODEL_ID,
    messages=function_request["messages"],
    inferenceConfig=function_request["inferenceConfig"],
    toolConfig=weather_tool
)
print(json.dumps(response, indent=2))

def handle_function_calling(model_id, request, tool_config):
    try:
        # Step 1: Send initial request
        response = bedrock.converse(
            modelId=model_id,
            messages=request["messages"],
            inferenceConfig=request["inferenceConfig"],
            toolConfig=tool_config
        )
        
        # Check if the model wants to use a tool (check the correct response structure)
        content_blocks = response["output"]["message"]["content"]
        has_tool_use = any("toolUse" in block for block in content_blocks)
        
        if has_tool_use:
            # Find the toolUse block
            tool_use_block = next(block for block in content_blocks if "toolUse" in block)
            tool_use = tool_use_block["toolUse"]
            tool_name = tool_use["name"]
            tool_input = tool_use["input"]
            tool_use_id = tool_use["toolUseId"]
            
            # Step 2: Execute the tool
            if tool_name == "get_weather":
                tool_result = get_weather(tool_input["location"])
            else:
                tool_result = {"error": f"Unknown tool: {tool_name}"}
            
            # Step 3: Send the tool result back to the model
            updated_messages = request["messages"] + [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": tool_use_id,
                                "name": tool_name,
                                "input": tool_input
                            }
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": tool_use_id,
                                "content": [
                                    {
                                        "json": tool_result
                                    }
                                ],
                                "status": "success"
                            }
                        }
                    ]
                }
            ]
            
            # Step 4: Get final response
            final_response = bedrock.converse(
                modelId=model_id,
                messages=updated_messages,
                inferenceConfig=request["inferenceConfig"],
                toolConfig=tool_config  
            )
            
            # Extract text from the correct response structure
            final_text = ""
            for block in final_response["output"]["message"]["content"]:
                if "text" in block:
                    final_text = block["text"]
                    break
            
            return {
                "tool_call": {"name": tool_name, "input": tool_input},
                "tool_result": tool_result,
                "final_response": final_text
            }
        else:
            # Model didn't use a tool, just return the text response
            text_response = ""
            for block in content_blocks:
                if "text" in block:
                    text_response = block["text"]
                    break
                    
            return {
                "final_response": text_response
            }
    
    except Exception as e:
        print(f"Error in function calling: {str(e)}")
        return {"error": str(e)}
    
function_result = handle_function_calling(
    MODEL_ID, 
    function_request,
    weather_tool
)

def display_response(response, model_name=None):
    if model_name:
        print(f"\n### Response from {model_name}\n")
    print(response)
    print("\n" + "-"*80 + "\n")
    
# Display the results
if "error" not in function_result:
    if "tool_call" in function_result:
        print(f"Tool Call: {function_result['tool_call']['name']}({function_result['tool_call']['input']})")
        print(f"Tool Result: {function_result['tool_result']}")
    
    display_response(function_result["final_response"], "Claude 3.7 Sonnet (Function Calling)")
else:
    print(f"Error: {function_result['error']}")
