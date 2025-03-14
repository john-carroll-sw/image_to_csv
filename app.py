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
from utils import setup_clients  # Updated import
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
logger.info(f"AZURE_GPT_4o_OPENAI_ENDPOINT: {os.getenv('AZURE_GPT_4o_OPENAI_ENDPOINT', 'Not set')[:10]}...")

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

def analyze_image_with_standard_output(client, deployment, image_bytes, model_info=None):
    """Fallback method using standard completion API with model-specific parameters."""
    base64_image = encode_image(image_bytes)
    
    # Default parameters for the API call
    params = {
        "model": deployment,
        "messages": [
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
    }
    
    # Check if we need to use max_completion_tokens instead of max_tokens
    if model_info and model_info.get("uses_max_completion_tokens", False):
        params["max_completion_tokens"] = 4000
        logger.info(f"Using max_completion_tokens for {model_info.get('model_name', 'o1')} model")
    else:
        params["max_tokens"] = 4000
        logger.info("Using max_tokens parameter (default)")
    
    response = client.chat.completions.create(**params)
    
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
    """Convert text response to DataFrame with advanced CSV parsing to handle mismatched columns."""
    try:
        # First try pandas read_csv with more flexible parameters
        try:
            # Try with standard pandas read_csv and error_bad_lines=False (allows skipping problematic rows)
            return pd.read_csv(io.StringIO(text_content), on_bad_lines='warn')
        except (pd.errors.ParserError, ValueError) as e:
            logger.warning(f"Standard CSV parsing failed: {str(e)}")
            
            # Second attempt with csv module for more control
            import csv
            
            # Reset the StringIO object
            csv_data = io.StringIO(text_content)
            
            # Use csv.Sniffer to detect delimiter and other format details
            try:
                dialect = csv.Sniffer().sniff(csv_data.read(1024))
                csv_data.seek(0)  # Reset after sniffing
                delimiter = dialect.delimiter
            except:
                # Default to comma if sniffing fails
                delimiter = ','
                csv_data.seek(0)  # Reset after failed sniff
            
            # Read all lines
            lines = list(csv.reader(csv_data, delimiter=delimiter))
            
            if not lines:
                raise ValueError("No data found in the CSV content")
                
            # Get header from first line
            header = lines[0]
            
            # Track if we needed to add columns
            columns_added = False
            max_cols = len(header)
            
            # First pass: find the maximum number of columns in any row
            for i, row in enumerate(lines[1:], 1):
                max_cols = max(max_cols, len(row))
            
            # Extend header if needed
            if max_cols > len(header):
                columns_added = True
                # Add missing header columns
                for i in range(len(header), max_cols):
                    header.append(f"Extra_Column_{i+1}")
                logger.info(f"Added {max_cols - len(header)} extra header columns to match longest row")
            
            # Second pass: ensure all rows have the right number of columns
            rows = []
            for i, row in enumerate(lines[1:], 1):
                if len(row) != len(header):
                    if len(row) < len(header):
                        # Pad shorter rows
                        padded_row = row + [''] * (len(header) - len(row))
                        rows.append(padded_row)
                        logger.info(f"Row {i}: Padded row with {len(header) - len(row)} empty values")
                    else:
                        # Truncate longer rows
                        rows.append(row[:len(header)])
                        logger.warning(f"Row {i}: Truncated row from {len(row)} to {len(header)} columns")
                else:
                    rows.append(row)
            
            # Create DataFrame with fixed data
            df = pd.DataFrame(rows, columns=header)
            
            # Show warning in the UI if we had to fix the data
            if columns_added:
                st.warning("⚠️ The CSV data had inconsistent column counts. Extra columns were added to the header.")
                
            return df
    except Exception as e:
        logger.error(f"CSV parsing failed with error: {str(e)}")
        st.error(f"Failed to parse CSV data: {str(e)}")
        
        # As a last resort, try returning the raw text split by lines
        try:
            lines = text_content.strip().split('\n')
            if len(lines) > 1:
                # Try to split by the most common delimiter found in the first line
                potential_delimiters = [',', '\t', ';', '|']
                first_line = lines[0]
                
                # Count delimiters in first line
                counts = {d: first_line.count(d) for d in potential_delimiters}
                best_delimiter = max(counts.items(), key=lambda x: x[1])[0]
                
                # Parse using the best delimiter
                rows = [line.split(best_delimiter) for line in lines]
                header = rows[0]
                
                # Same padding logic as before
                max_cols = max(len(row) for row in rows)
                
                # Extend header if needed
                if max_cols > len(header):
                    header.extend([f"Column_{i+1}" for i in range(len(header), max_cols)])
                
                # Create DataFrame with fixed data
                return pd.DataFrame([row + [''] * (len(header) - len(row)) for row in rows[1:]], columns=header)
        except:
            pass
            
        # Return the raw text for manual handling
        st.info("Couldn't parse as CSV. Showing raw text for manual processing.")
        return text_content

def reset_prompts():
    """Reset prompts and schema to default values"""
    st.session_state.structured_system_prompt = DEFAULT_STRUCTURED_SYSTEM_PROMPT
    st.session_state.structured_user_prompt = DEFAULT_STRUCTURED_USER_PROMPT
    st.session_state.standard_system_prompt = DEFAULT_STANDARD_SYSTEM_PROMPT
    st.session_state.standard_user_prompt = DEFAULT_STANDARD_USER_PROMPT
    st.session_state.schema_json = DEFAULT_SCHEMA_JSON

def analyze_image_with_o1_and_gpt4o_processing(client_o1, deployment_o1, client_gpt4o, deployment_gpt4o, image_bytes):
    """Two-stage processing: o1 for image analysis, GPT-4o for structured formatting."""
    # Step 1: Process the image with o1 to extract the raw data
    base64_image = encode_image(image_bytes)
    
    # Use o1 to extract text content from the image using max_completion_tokens
    o1_response = client_o1.chat.completions.create(
        model=deployment_o1,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in analyzing images and extracting data. Extract all tabular or structured data visible in the image as plain text, preserving formatting and relationships between data items."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all tabular data from this image exactly as shown. Include all rows, columns, and values."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        max_completion_tokens=40000
    )
    
    o1_extracted_text = o1_response.choices[0].message.content
    logger.info(f"o1 extracted text: {o1_extracted_text}...")
    
    # Step 2: Send the extracted text to GPT-4o for structured output processing
    response_format = create_pydantic_model_from_json(st.session_state.schema_json)
    
    try:
        # Send the text extracted by o1 to GPT-4o for structured formatting
        gpt4o_response = client_gpt4o.beta.chat.completions.parse(
            model=deployment_gpt4o,
            messages=[
                {
                    "role": "system",
                    "content": st.session_state.structured_system_prompt
                },
                {
                    "role": "user",
                    "content": f"Format this extracted data into a CSV structure with headers and rows:\n\n{o1_extracted_text}"
                }
            ],
            temperature=0,
            response_format=response_format
        )
        
        # Return both the o1 raw extraction and the GPT-4o structured output
        return {
            "structured_result": gpt4o_response.choices[0].message.parsed,
            "raw_extraction": o1_extracted_text
        }
        
    except Exception as e:
        st.error(f"Error with GPT-4o structured formatting: {str(e)}")
        # Return just the raw o1 extraction which will be handled as a fallback
        return {
            "structured_result": None,
            "raw_extraction": o1_extracted_text
        }

def main():
    # Setup clients for all models
    models = setup_clients()
    
    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], 
                                   help="Upload a screenshot of a table or structured data")
    
    # Model selection - now includes o1 if available
    available_models = [model for model in models if models[model] is not None]
    model_option = st.selectbox(
        "Select model",
        available_models,
        index=0,
        help="Select the Azure OpenAI model to use for image analysis"
    )
    
    # Display model information
    selected_model = models[model_option]
    
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
    
    # Get the client and deployment for the selected model
    selected_model = models[model_option]
    client = selected_model["client"]
    deployment = selected_model["deployment"]
    supports_structured = selected_model.get("supports_structured", True)  # Default to True for backward compatibility
    
    if uploaded_file is not None:
        # Display the uploaded image using our safe wrapper
        safe_st_image(uploaded_file, caption="Uploaded Image")
        
        # Process when button is clicked
        if st.button("Convert to CSV"):
            try:
                with st.spinner(f"Analyzing image with {model_option}..."):
                    # Get image bytes
                    image_bytes = uploaded_file.getvalue()
                    
                    # Get GPT-4o client for formatting regardless of which model is selected for image analysis
                    gpt4o_info = models["GPT-4o"]
                    
                    if model_option == "o1":
                        # Use the two-stage approach: o1 for extraction, GPT-4o for formatting
                        st.info(f"Using {model_option} to analyze the image, then GPT-4o to format the data")
                        
                        o1_info = models[model_option]
                        result = analyze_image_with_o1_and_gpt4o_processing(
                            o1_info["client"],
                            o1_info["deployment"],
                            gpt4o_info["client"],
                            gpt4o_info["deployment"],
                            image_bytes
                        )
                        
                        structured_result = result["structured_result"]
                        
                        # Always show the raw extraction from o1
                        with st.expander("Raw o1 Extraction", expanded=False):
                            st.text_area("Raw Text", result["raw_extraction"], height=200)
                        
                    else:
                        # Standard approach for GPT-4o
                        supports_structured = selected_model.get("supports_structured", True)
                        
                        # Only try structured output if the model supports it
                        structured_result = None
                        if supports_structured:
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
                        with st.spinner("Using standard method for CSV extraction..."):
                            if model_option == "o1" and "raw_extraction" in locals():
                                # For o1, we already have the raw extraction
                                standard_result = result["raw_extraction"]
                            else:
                                # For other models, get the standard output
                                standard_result = analyze_image_with_standard_output(
                                    client,
                                    deployment, 
                                    image_bytes,
                                    model_info=selected_model
                                )
                            
                            # Show which method was used
                            st.info("🔄 Used Standard Output Mode")
                            
                            # ALWAYS display the raw output
                            st.subheader("Raw API Response:")
                            st.text_area("CSV Text (Raw)", standard_result, height=200)
                            
                            # Try to convert to DataFrame
                            result_data = convert_standard_to_dataframe(standard_result)
                            
                            if isinstance(result_data, pd.DataFrame):
                                st.success("Successfully converted to CSV!")
                                
                                # Show warning if we had to fix columns
                                if any(col.startswith('Column') for col in result_data.columns):
                                    st.warning("⚠️ Column headers were auto-generated to fix mismatched data. You may want to adjust them manually.")
                                
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
