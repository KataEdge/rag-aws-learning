import boto3
from langchain_aws import BedrockLLM

def get_bedrock_llm():
    """
    Get the Bedrock LLM.
    """
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
    )
    
    llm = BedrockLLM(
        client=bedrock_runtime,
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    )
    
    return llm