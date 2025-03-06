# FROM python:3.10-slim
FROM python:3.12-slim

WORKDIR /app

# Install dependencies with exact versions from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Print installed package versions for debugging
RUN pip list | grep -E "streamlit|pandas|openai|pydantic"
RUN pip show streamlit | grep -E "Version|Location"

# Copy application code
COPY . .

# Set environment variable for Streamlit
ENV STREAMLIT_SERVER_PORT=8000
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Default entry point is app.py but can be overridden
ARG ENTRY_FILE=app.py
ENV ENTRY_FILE=${ENTRY_FILE}

# Expose port
EXPOSE 8000

# Start the Streamlit app with additional debug information
CMD echo "Starting Streamlit app with Python $(python --version)" && \
    streamlit --version && \
    streamlit run $ENTRY_FILE --server.port 8000 --server.enableCORS false --server.enableXsrfProtection false
