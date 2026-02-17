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


# Now create multi-turn messages using the Mistral response
# Call Claude with multi-turn conversation
# Example of streaming with Converse API
def stream_converse(model_id, messages, inference_config=None):
    if inference_config is None:
        inference_config = {}
streaming_request = [
    {
        "role": "user",
        "content": [
            {
                "text": f"""Please provide a detailed summary of the following text, explaining its key points and implications:
                
                {text_to_summarize}
                
                Make your summary comprehensive but clear.
                """
            }
        ]
    }
]

# Call Claude with multi-turn conversation
# Example of streaming with Converse API
def stream_converse(model_id, messages, inference_config=None):
    if inference_config is None:
        inference_config = {}
    
    print("Streaming response (chunks will appear as they are received):\n")
    print("-" * 80)
    
    full_response = ""
    
    try:
        response = bedrock.converse_stream(
            modelId=model_id,
            messages=messages,
            inferenceConfig=inference_config
        )
        response_stream = response.get('stream')
        if response_stream:
            for event in response_stream:

                if 'messageStart' in event:
                    print(f"\nRole: {event['messageStart']['role']}")

                if 'contentBlockDelta' in event:
                    print(event['contentBlockDelta']['delta']['text'], end="")

                if 'messageStop' in event:
                    print(f"\nStop reason: {event['messageStop']['stopReason']}")

                if 'metadata' in event:
                    metadata = event['metadata']
                    if 'usage' in metadata:
                        print("\nToken usage")
                        print(f"Input tokens: {metadata['usage']['inputTokens']}")
                        print(
                            f":Output tokens: {metadata['usage']['outputTokens']}")
                        print(f":Total tokens: {metadata['usage']['totalTokens']}")
                    if 'metrics' in event['metadata']:
                        print(
                            f"Latency: {metadata['metrics']['latencyMs']} milliseconds")

                
            print("\n" + "-" * 80)
        return full_response
    
    except Exception as e:
        print(f"Error in streaming: {str(e)}")
        return None
    
streamed_response = stream_converse(
    MODEL_ID, 
    streaming_request, 
    inference_config={"temperature": 0.4, "maxTokens": 1000}
)