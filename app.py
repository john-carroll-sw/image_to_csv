import base64
import io
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from utils import setup_client

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
    st.title("Image to CSV Converter")
    st.write("Upload an image and let AI convert it to CSV format.")
    
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
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
        
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
                        st.dataframe(df, use_container_width=True)
                        
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
                                st.dataframe(result_data, use_container_width=True)
                                
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
