import base64
import io
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from utils import setup_client
from auth import require_auth, get_username, logout, is_authenticated, AUTH_ENABLED

# Set page configuration
st.set_page_config(
    page_title="Image to CSV Converter",
    page_icon="📊",
)

# Check for logout action through query parameter
if AUTH_ENABLED and "logout" in st.query_params:
    st.query_params.clear()
    logout()
    # Don't call st.rerun() here, logout() will handle redirection

# Check authentication before proceeding
if not require_auth():
    st.stop()

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
    
    try:
        response = client.beta.chat.completions.parse(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in analyzing images and extracting structured data. Extract data from the image into a CSV format with appropriate headers and data values. Ensure all rows have the same number of columns as there are headers."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please analyze this image and provide the data in a structured format with headers and rows."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0,
            response_format=CSVData
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
                "content": "You are an expert in analyzing images and extracting structured data. Extract data from the image into a CSV format with appropriate headers and data types. Return ONLY the CSV data without any markdown formatting or explanation text."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please analyze this image and provide the data in a CSV format."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        max_tokens=4000,
    )
    
    return response.choices[0].message.content

def convert_structured_to_dataframe(csv_data):
    """Convert structured CSV data to pandas DataFrame."""
    headers = csv_data.headers
    rows = csv_data.rows
    
    # Create DataFrame
    return pd.DataFrame(rows, columns=headers)

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
        help="GPT-4o offers higher accuracy but may be slower"
    )
    
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
                        # Convert structured output to DataFrame
                        st.success("✅ Successfully extracted structured data!")
                        
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
