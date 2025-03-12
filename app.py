import base64
import io
import os
import json
import streamlit as st
import pandas as pd
import logging
import sys
from dotenv import load_dotenv
from pydantic import BaseModel, create_model
from typing import List, Optional, Dict, Any
from utils import setup_client
from auth import require_auth, logout, is_authenticated, AUTH_ENABLED

# Configure logging to stdout for Azure Web App logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - APP - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables and log key variables
load_dotenv(override=True)
logger.info(f"AUTH_ENABLED: {os.getenv('VITE_AUTH_ENABLED', 'Not set')}")
logger.info(f"FRONTEND_URL: {os.getenv('FRONTEND_URL', 'Not set')}")
logger.info(f"AUTH_URL: {os.getenv('VITE_AUTH_URL', 'Not set')}")
logger.info(f"AZURE_OPENAI_EASTUS_ENDPOINT: {os.getenv('AZURE_OPENAI_EASTUS_ENDPOINT', 'Not set')[:10]}...")

# Set page configuration
st.set_page_config(
    page_title="Image to CSV Converter",
    page_icon="📊",
)

logger.info("Starting authentication process")

# Check for logout action through query parameter
if AUTH_ENABLED and "logout" in st.query_params:
    logger.info("Logout action detected in query parameters")
    st.query_params.clear()
    logout()
    # Don't call st.rerun() here, logout() will handle redirection

# Check authentication before proceeding
logger.info("Checking authentication status")
if not require_auth():
    logger.info("Authentication failed, stopping app execution")
    st.stop()

logger.info("Authentication successful, continuing with app")

# # Debug environment variables (remove this in production)
# if AUTH_ENABLED:
#     with st.expander("Debug Environment", expanded=False):
#         st.write(f"FRONTEND_URL: {os.environ.get('FRONTEND_URL', 'Not set')}")
#         st.write(f"Auth URL: {os.environ.get('VITE_AUTH_URL', 'Not set')}")
#         st.write(f"Auth Enabled: {os.environ.get('VITE_AUTH_ENABLED', 'Not set')}")
#         from auth import FRONTEND_INFO
#         st.write(f"FRONTEND_INFO in auth.py: {FRONTEND_INFO}")

# Check for logout action through query parameter
if AUTH_ENABLED and "logout" in st.query_params:
    st.query_params.clear()
    logout()
    # Don't call st.rerun() here, logout() will handle redirection

# Check authentication before proceeding
if not require_auth():
    st.stop()

# Define default prompt templates and schema
DEFAULT_STRUCTURED_SYSTEM_PROMPT = """You are an expert in analyzing images and extracting structured data. Extract data from the image into a CSV format with appropriate headers and data values. Ensure all rows have the same number of columns as there are headers."""

DEFAULT_STRUCTURED_USER_PROMPT = """Please analyze this image and provide the data in a structured format with headers and rows."""

DEFAULT_STANDARD_SYSTEM_PROMPT = """You are an expert in analyzing images and extracting structured data. Extract data from the image into a CSV format with appropriate headers and data types. Return ONLY the CSV data without any markdown formatting or explanation text."""

DEFAULT_STANDARD_USER_PROMPT = """Please analyze this image and provide the data in a CSV format."""

# Define default Pydantic schema as a JSON string
DEFAULT_SCHEMA_JSON = """
{
    "CSVData": {
        "headers": "List[str]",
        "rows": "List[List[str]]"
    }
}
"""

# Initialize session state for editable prompts and schema
if "structured_system_prompt" not in st.session_state:
    st.session_state.structured_system_prompt = DEFAULT_STRUCTURED_SYSTEM_PROMPT
if "structured_user_prompt" not in st.session_state:
    st.session_state.structured_user_prompt = DEFAULT_STRUCTURED_USER_PROMPT
if "standard_system_prompt" not in st.session_state:
    st.session_state.standard_system_prompt = DEFAULT_STANDARD_SYSTEM_PROMPT
if "standard_user_prompt" not in st.session_state:
    st.session_state.standard_user_prompt = DEFAULT_STANDARD_USER_PROMPT
if "schema_json" not in st.session_state:
    st.session_state.schema_json = DEFAULT_SCHEMA_JSON

# Header with GitHub link and user info
header_cols = st.columns([4, 2])
with header_cols[0]:
    # GitHub repository link
    st.markdown(
        """
        <div style="margin-top: 5px;">
            <a href="https://github.com/john-carroll-sw/image_to_csv" target="_blank">
                <img src="https://img.shields.io/badge/GitHub-View%20on%20GitHub-blue?logo=github" alt="GitHub Repository"/>
            </a>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
with header_cols[1]:
    # Only show user info if authenticated
    if AUTH_ENABLED and is_authenticated():
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; align-items: center;">
            <a href="?logout=true" style="display: inline-flex; align-items: center; justify-content: center; 
                padding: 5px 12px; border-radius: 4px; background: transparent; 
                color: currentColor; border: none; cursor: pointer; transition: background 0.2s;
                text-decoration: none;" 
                onmouseover="this.style.background='rgba(0,0,0,0.05)'" 
                onmouseout="this.style.background='transparent'">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" 
            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
            style="margin-right: 5px;">
            <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
            <polyline points="16 17 21 12 16 7"></polyline>
            <line x1="21" y1="12" x2="9" y2="12"></line>
                </svg>
                Logout
            </a>
        </div>
        """, unsafe_allow_html=True)

# Full width title below
st.title("Image to CSV Converter")

st.write("Upload an image and let AI convert it to CSV format.")

# Define simple wrapper functions that handle version differences directly
def safe_st_image(image, caption=None):
    """Display an image in a way that works across Streamlit versions."""
    try:
        # Try the newer parameter name first
        return st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        # Fall back to the older parameter name
        return st.image(image, caption=caption, use_column_width=True)

def safe_st_dataframe(data):
    """Display a dataframe in a way that works across Streamlit versions."""
    try:
        # Try the newer parameter name first
        return st.dataframe(data, use_container_width=True)
    except TypeError:
        try:
            # Try the middle version parameter name
            return st.dataframe(data, width=None)
        except:
            # Just display it without width parameters
            return st.dataframe(data)

def parse_type_annotation(annotation_str):
    """Parse a string type annotation into a Python type."""
    type_mapping = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "List[str]": List[str],
        "List[int]": List[int],
        "List[float]": List[float],
        "List[List[str]]": List[List[str]],
        "List[List[int]]": List[List[int]],
        "List[List[float]]": List[List[float]],
        "Dict[str, Any]": Dict[str, Any],
        "Optional[str]": Optional[str],
        "Any": Any
    }
    
    return type_mapping.get(annotation_str.strip(), str)

def create_pydantic_model_from_json(schema_json):
    """Dynamically create a Pydantic model from a JSON schema definition."""
    try:
        schema_dict = json.loads(schema_json)
        
        # We expect a dictionary with a single key (model name) and a nested dictionary (fields)
        model_name = list(schema_dict.keys())[0]
        fields = {}
        
        for field_name, field_type_str in schema_dict[model_name].items():
            fields[field_name] = (parse_type_annotation(field_type_str), ...)
        
        # Create and return the model
        return create_model(model_name, **fields)
    
    except Exception as e:
        st.error(f"Error creating model from schema: {str(e)}")
        
        # Fall back to default model
        class CSVData(BaseModel):
            headers: List[str]
            rows: List[List[str]]
        
        return CSVData

# Define the structured output model
class CSVRow(BaseModel):
    values: List[str]

class CSVData(BaseModel):
    headers: List[str]
    rows: List[List[str]]

def encode_image(image_bytes):
    """Encode image to base64."""
    return base64.b64encode(image_bytes).decode('utf-8')

def analyze_image_with_structured_output(client, deployment, image_bytes):
    """Send image to Azure OpenAI and get structured CSV data."""
    base64_image = encode_image(image_bytes)
    
    # Create Pydantic model from schema
    response_format = create_pydantic_model_from_json(st.session_state.schema_json)
    
    try:
        response = client.beta.chat.completions.parse(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": st.session_state.structured_system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": st.session_state.structured_user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0,
            response_format=response_format
        )
        
        return response.choices[0].message.parsed
    except Exception as e:
        st.error(f"Error with structured output: {str(e)}")
        # Fall back to standard completion if structured output fails
        return None

def analyze_image_with_standard_output(client, deployment, image_bytes):
    """Fallback method using standard completion API."""
    base64_image = encode_image(image_bytes)
    
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": st.session_state.standard_system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": st.session_state.standard_user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        max_tokens=4000,
    )
    
    return response.choices[0].message.content

def convert_structured_to_dataframe(structured_data):
    """Convert structured data to pandas DataFrame."""
    # Handle different schema formats
    if hasattr(structured_data, 'headers') and hasattr(structured_data, 'rows'):
        # Standard CSV format
        headers = structured_data.headers
        rows = structured_data.rows
        return pd.DataFrame(rows, columns=headers)
    elif hasattr(structured_data, 'data') and isinstance(structured_data.data, list):
        # List of dictionaries format
        return pd.DataFrame(structured_data.data)
    else:
        # Try to convert the object to a dict and then to DataFrame
        try:
            data_dict = structured_data.dict()
            if isinstance(data_dict, dict):
                # If it's a flat dictionary, convert to a single-row DataFrame
                if not any(isinstance(v, (list, dict)) for v in data_dict.values()):
                    return pd.DataFrame([data_dict])
                # If it contains a 'rows' or 'data' key that is a list
                elif 'rows' in data_dict and isinstance(data_dict['rows'], list):
                    if 'headers' in data_dict:
                        return pd.DataFrame(data_dict['rows'], columns=data_dict['headers'])
                    else:
                        return pd.DataFrame(data_dict['rows'])
                elif 'data' in data_dict and isinstance(data_dict['data'], list):
                    return pd.DataFrame(data_dict['data'])
                else:
                    return pd.DataFrame.from_dict(data_dict)
        except Exception as e:
            st.warning(f"Conversion to DataFrame failed: {str(e)}")
            return pd.DataFrame([vars(structured_data)])

def convert_standard_to_dataframe(text_content):
    """Convert text response to DataFrame as a fallback."""
    try:
        # Use StringIO to create a file-like object from the string
        csv_data = io.StringIO(text_content)
        return pd.read_csv(csv_data)
    except Exception as e:
        st.warning(f"Standard parsing failed: {str(e)}")
        
        # Return the raw text for manual handling
        return text_content

def reset_prompts():
    """Reset prompts and schema to default values"""
    st.session_state.structured_system_prompt = DEFAULT_STRUCTURED_SYSTEM_PROMPT
    st.session_state.structured_user_prompt = DEFAULT_STRUCTURED_USER_PROMPT
    st.session_state.standard_system_prompt = DEFAULT_STANDARD_SYSTEM_PROMPT
    st.session_state.standard_user_prompt = DEFAULT_STANDARD_USER_PROMPT
    st.session_state.schema_json = DEFAULT_SCHEMA_JSON

def main():
    # Setup client
    client, gpt4o_deployment = setup_client()
    
    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], 
                                   help="Upload a screenshot of a table or structured data")
    
    # Model selection
    model_option = st.selectbox(
        "Select model",
        ["GPT-4o"],
        index=0,
        help="WIP: May add more models in the future"
    )
    
    # Create an expander for configuration (renamed from "Edit Model Prompts")
    with st.expander("Configuration", expanded=False):
        # Add a reset button at the top
        if st.button("Reset to Defaults"):
            reset_prompts()
            st.rerun()
        
        st.markdown("### Structured Output Mode")
        st.markdown("#### System Prompt:")
        # Fix: Use a different key for the widget than the session state variable
        structured_sys = st.text_area(
            "Structured System Prompt", 
            value=st.session_state.structured_system_prompt,
            height=150, 
            key="structured_system_input"
        )
        # Update session state separately
        st.session_state.structured_system_prompt = structured_sys
        
        st.markdown("#### User Prompt:")
        structured_user = st.text_area(
            "Structured User Prompt", 
            value=st.session_state.structured_user_prompt,
            height=100, 
            key="structured_user_input"
        )
        st.session_state.structured_user_prompt = structured_user
        
        st.markdown("#### Response Schema (JSON):")
        st.info("Define the structure of the expected response. This must be a valid JSON with type annotations.")
        schema = st.text_area(
            "Schema Definition", 
            value=st.session_state.schema_json,
            height=150, 
            key="schema_json_input"
        )
        st.session_state.schema_json = schema
        
        # Schema examples as part of the main expander
        st.markdown("#### Schema Examples:")
        st.markdown("Standard CSV format:")
        st.code("""
{
    "CSVData": {
        "headers": "List[str]",
        "rows": "List[List[str]]"
    }
}
        """, language="json")
        
        st.markdown("Table with list of records format:")
        st.code("""
{
    "TableData": {
        "data": "List[Dict[str, Any]]"
    }
}
        """, language="json")
        
        # Checkbox to show/hide fallback configuration
        show_fallback = st.checkbox("Show Fallback Configuration", value=False, 
                                   help="Configure the fallback mode that runs if structured output fails")
        
        if show_fallback:
            st.markdown("---")
            st.markdown("### Standard Output Mode (Fallback)")
            st.markdown("""
            <div style="padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <p><strong>Note:</strong> This fallback mode is used when structured output parsing fails. 
                It requests raw CSV text and attempts to parse it directly.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### System Prompt:")
            standard_sys = st.text_area(
                "Standard System Prompt", 
                value=st.session_state.standard_system_prompt,
                height=150, 
                key="standard_system_input"
            )
            st.session_state.standard_system_prompt = standard_sys
            
            st.markdown("#### User Prompt:")
            standard_user = st.text_area(
                "Standard User Prompt", 
                value=st.session_state.standard_user_prompt,
                height=100, 
                key="standard_user_input"
            )
            st.session_state.standard_user_prompt = standard_user
    
    # Map selection to deployment name
    deployment = gpt4o_deployment if model_option == "GPT-4o" else "GPT-4o"
    
    if uploaded_file is not None:
        # Display the uploaded image using our safe wrapper
        safe_st_image(uploaded_file, caption="Uploaded Image")
        
        # Process when button is clicked
        if st.button("Convert to CSV"):
            try:
                with st.spinner(f"Analyzing image with {model_option}..."):
                    # Get image bytes
                    image_bytes = uploaded_file.getvalue()
                    
                    # First try with structured outputs
                    structured_result = analyze_image_with_structured_output(client, deployment, image_bytes)
                    
                    if structured_result:
                        # Show which method was used
                        st.info("✅ Used Structured Output Mode")
                        
                        # Convert structured output to DataFrame
                        st.success("Successfully extracted structured data!")
                        
                        # Convert to DataFrame and display
                        df = convert_structured_to_dataframe(structured_result)
                        st.subheader("CSV Output:")
                        
                        # Use safe wrapper for dataframe
                        safe_st_dataframe(df)
                        
                        # Generate CSV for download
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"{uploaded_file.name.split('.')[0]}_data.csv",
                            mime="text/csv",
                            key="download-csv"
                        )
                    else:
                        # Fall back to standard output
                        with st.spinner("Structured parsing failed, trying standard method..."):
                            standard_result = analyze_image_with_standard_output(client, deployment, image_bytes)
                            
                            # Show which method was used
                            st.info("🔄 Used Standard Output Mode (Fallback)")
                            
                            # Display raw output in expander
                            with st.expander("Show Raw API Response", expanded=False):
                                st.text_area("Raw Response", standard_result, height=200)
                            
                            # Try to convert to DataFrame
                            result_data = convert_standard_to_dataframe(standard_result)
                            
                            if isinstance(result_data, pd.DataFrame):
                                st.success("Successfully converted to CSV!")
                                st.subheader("CSV Output:")
                                
                                # Use safe wrapper for dataframe
                                safe_st_dataframe(result_data)
                                
                                # Generate CSV for download
                                csv = result_data.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download CSV",
                                    data=csv,
                                    file_name=f"{uploaded_file.name.split('.')[0]}_data.csv",
                                    mime="text/csv",
                                    key="download-csv"
                                )
                            else:
                                st.text_area("CSV Text (Copy and paste into a .csv file)", result_data, height=300)
                                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                
    # Add a footer with instructions
    st.divider()
    st.caption("How to use: Upload an image containing a table or structured data, select a model, and click 'Convert to CSV'. The application will extract the data and provide a downloadable CSV file.")

if __name__ == "__main__":
    main()
