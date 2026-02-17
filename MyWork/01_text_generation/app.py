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

MODEL_ID = "mistral.mistral-large-2402-v1:0" # Add this line

MODELS = {
    "Claude 3.7 Sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "Claude 3.5 Sonnet": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "Claude 3.5 Haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "Amazon Nova Pro": "us.amazon.nova-pro-v1:0",
    "Amazon Nova Micro": "us.amazon.nova-micro-v1:0",
    "Meta Llama 3.1 70B Instruct": "us.meta.llama3-1-70b-instruct-v1:0",
    "Magistral Small 2509": "mistral.magistral-small-2509"
}

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

# Mistral uses a simpler prompt format
request_body = json.dumps({
    "prompt": prompt,
    "max_tokens": 1000,
    "temperature": 0.1,
    "top_p": 0.1,
})

try:
    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=request_body,
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response.get('body').read())
    
    # Extract text from Mistral response
    result_text = response_body.get("outputs", [{}])[0].get("text", "")
    display_response(result_text, "Magistral Small 2509")

except botocore.exceptions.ClientError as error:
    if error.response['Error']['Code'] == 'AccessDeniedException':
        print(f"\x1b[41m{error.response['Error']['Message']}\
            \nTroubleshoot: https://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\x1b[0m\n")
    else:
        raise error