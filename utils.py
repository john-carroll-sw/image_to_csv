import os
from dotenv import load_dotenv
from openai import AzureOpenAI

def setup_client():
    """Set up and return the Azure OpenAI client and model deployment name."""
    load_dotenv()
    
    endpoint = os.getenv("AZURE_OPENAI_EASTUS_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_EASTUS_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    gpt4o_deployment = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT")
    
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version
    )
    
    return client, gpt4o_deployment
