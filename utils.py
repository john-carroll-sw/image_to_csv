import os
from dotenv import load_dotenv
from openai import AzureOpenAI

def setup_clients():
    """Set up and return Azure OpenAI clients and model deployment names for all models."""
    load_dotenv()
    
    # GPT-4o client setup
    gpt4o_endpoint = os.getenv("AZURE_GPT_4o_OPENAI_ENDPOINT")
    gpt4o_api_key = os.getenv("AZURE_GPT_4o_OPENAI_API_KEY")
    gpt4o_api_version = os.getenv("AZURE_GPT_4o_OPENAI_API_VERSION", "2024-08-01-preview")
    gpt4o_deployment = os.getenv("AZURE_GPT_4o_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    
    # o1 client setup
    o1_endpoint = os.getenv("AZURE_o1_OPENAI_ENDPOINT")
    o1_api_key = os.getenv("AZURE_o1_OPENAI_API_KEY")
    o1_api_version = os.getenv("AZURE_o1_OPENAI_API_VERSION", "2024-12-17")
    o1_deployment = os.getenv("AZURE_o1_OPENAI_DEPLOYMENT_NAME", "o1")
    
    # Create clients
    gpt4o_client = AzureOpenAI(
        azure_endpoint=gpt4o_endpoint,
        api_key=gpt4o_api_key,
        api_version=gpt4o_api_version
    )
    
    # Check if o1 credentials are available
    o1_client = None
    if o1_endpoint and o1_api_key:
        o1_client = AzureOpenAI(
            azure_endpoint=o1_endpoint,
            api_key=o1_api_key,
            api_version=o1_api_version
        )
    
    # Create a model mapping for easy access - include API version info for reference
    models = {
        "GPT-4o": {
            "client": gpt4o_client, 
            "deployment": gpt4o_deployment,
            "api_version": gpt4o_api_version,
            "supports_structured": True,
            "uses_max_completion_tokens": False,
            "model_name": "GPT-4o"
        },
        "o1": {
            "client": o1_client, 
            "deployment": o1_deployment,
            "api_version": o1_api_version,
            "supports_structured": False,  # Currently, we'll skip structured output for o1
            "uses_max_completion_tokens": True,  # o1 uses max_completion_tokens instead of max_tokens
            "model_name": "o1"
        } if o1_client else None
    }
    
    return models
