# 📋 Image to CSV Converter

A Streamlit application that leverages Azure OpenAI's vision capabilities to convert images containing tables or structured data into CSV format for easy use in spreadsheets and data analysis.

![Demo Screenshot](UI.png)

## ✨ Features

- 🖼️ Upload any image containing tabular data or structured text
- 🤖 Process with GPT-4o or other vision models
- 📊 Extract structured data with high accuracy using advanced AI
- 📥 Download results as ready-to-use CSV files
- 🔄 Fallback processing options if initial extraction fails

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Azure OpenAI API access with GPT-4o capability
- Azure subscription with vision features enabled

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image_to_csv.git
   cd image_to_csv
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your `.env` file with your Azure OpenAI credentials:
   ```bash
   # Copy the sample environment file and edit it with your credentials
   cp .env.sample .env
   # Now edit the .env file with your preferred editor
   nano .env  # or use any text editor you prefer
   ```

   Your `.env` file should contain:
   ```
   AZURE_OPENAI_EASTUS_ENDPOINT=your_endpoint_here
   AZURE_OPENAI_EASTUS_API_KEY=your_api_key_here
   AZURE_OPENAI_API_VERSION=2024-08-01-preview
   AZURE_OPENAI_GPT4O_DEPLOYMENT=gpt-4o
   ```

### Running the Application

Run the Streamlit application:
```bash
streamlit run app.py
```

Open your web browser to http://localhost:8501 to use the application.

## 📖 Usage

1. **Upload an Image**: Click the upload button to select an image file containing a table or structured data.

2. **Select Model**: Choose between GPT-4o (higher accuracy) or [add other vision capable models].

3. **Convert**: Click the "Convert to CSV" button to start the extraction process.

4. **Review and Download**: After processing, review the extracted data in the interactive table view, then download the CSV file using the download button.

## 🧰 How It Works

The application uses a multi-stage approach to extract data:

1. First, it attempts to use the OpenAI structured outputs feature to extract data in a precisely defined format.
   
2. If that fails, it falls back to a standard completion approach with specialized CSV parsing.

3. The resulting data is presented in an interactive table and made available for download as a CSV file.

## 📁 Project Structure

```
image_to_csv/
├── app.py               # Main Streamlit application
├── utils.py             # Utility functions for OpenAI client setup
├── requirements.txt     # Project dependencies
├── .env                 # Environment variables (not tracked in git)
├── .env.sample          # Sample environment variables template
├── README.md            # Project documentation
├── LICENSE              # MIT License
└── docs/                # Documentation assets
    └── images/          # Screenshots and images for documentation
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service/) for providing the powerful AI models
- [Streamlit](https://streamlit.io/) for the simple web application framework
- [Pandas](https://pandas.pydata.org/) for data handling capabilities
