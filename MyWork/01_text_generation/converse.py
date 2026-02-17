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

bedrock_control = boto3.client('bedrock', region_name=region, 
                                aws_access_key_id=KEY_ID,
                                aws_secret_access_key=ACCESS_KEY)
models = bedrock_control.list_foundation_models()
mistral_models = [m['modelId'] for m in models['modelSummaries'] if 'mistral' in m['modelId'].lower()]
print("Available Mistral models:", mistral_models)

MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0" # Add this line

def display_response(response, model_name=None):
    if model_name:
        print(f"\n### Response from {model_name}\n")
    print(response)
    print("\n" + "-"*80 + "\n")

text_to_summarize = """
AWS took all of that feedback from customers, and today we are excited to announce Amazon Bedrock, \
a new service that makes FMs from AI21 Labs, Anthropic, Stability AI, and Amazon accessible via an API. \
Bedrock is the easiest way for customers to build and scale generative AI-based applications using FMs, \
democratizing access for all builders. Bedrock will offer the ability to access a range of powerful FMs \
for text and images—including Amazons Titan FMs, which consist of two new LLMs we're also announcing \
today—through a scalable, reliable, and secure AWS managed service. With Bedrock's serverless experience, \
customers can easily find the right model for what they're trying to get done, get started quickly, privately \
customize FMs with their own data, and easily integrate and deploy them into their applications using the AWS \
tools and capabilities they are familiar with, without having to manage any infrastructure (including integrations \
with Amazon SageMaker ML features like Experiments to test different models and Pipelines to manage their FMs at scale).
"""

prompt = f"""Summarize the text below without adding information not present in it.

{text_to_summarize}
"""

converse_request = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "text": f"Please provide a concise summary of the following text in 2-3 sentences. Text to summarize: {text_to_summarize}"
                }
            ]
        }
    ],
    "inferenceConfig": {
        "temperature": 0.4,
        "topP": 0.9,
        "maxTokens": 500
    }
}

results = {}

# First, call Mistral to get initial summary
try:
    mistral_model = "mistral.mistral-7b-instruct-v0:2"
    mistral_response = bedrock.converse(
        modelId=mistral_model,
        messages=[{
            "role": "user",
            "content": [{"text": f"Please summarize this text: {text_to_summarize}"}]
        }],
        inferenceConfig=converse_request["inferenceConfig"]
    )
    
    mistral_summary = mistral_response["output"]["message"]["content"][0]["text"]
    results[mistral_model] = {"response": mistral_summary}
    display_response(mistral_summary, "Mistral 7B (Initial Summary)")
except botocore.exceptions.ClientError as error:
    print(f"Error calling Mistral: {error}")
    results[mistral_model] = {"response": "Summary not available"}

# Now create multi-turn messages using the Mistral response
multi_turn_messages = [
    {
        "role":"user",
        "content": [{"text": f"Please summarize this text: {text_to_summarize}"}]
    },
    {
        "role":"assistant",
        "content": [{"text": results["mistral.mistral-7b-instruct-v0:2"]["response"]}]
    },
    {
        "role":"user",
        "content": [{"text": "Can you make this summary even shorter, just 1 sentence?"}]
    }
]

# Call Claude with multi-turn conversation
try:
    multi_turn_response = bedrock.converse(
        modelId=MODEL_ID,
        messages=multi_turn_messages,
        inferenceConfig=converse_request["inferenceConfig"]
    )
    
    claude_multi_turn = multi_turn_response["output"]["message"]["content"][0]["text"]
    display_response(claude_multi_turn, "Claude 3 Haiku (Multi-turn - Shorter Summary)")
except botocore.exceptions.ClientError as error:
    print(f"Error in multi-turn conversation: {error}")