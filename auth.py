import os
import json
import time
import base64
import requests
import streamlit as st
import logging
from urllib.parse import urlencode
from dotenv import load_dotenv

# Set up logger
logger = logging.getLogger('auth')

# Load environment variables
load_dotenv(override=True)

# Get auth URL from environment variable - never hardcode in the source
AUTH_URL = os.getenv("VITE_AUTH_URL")
AUTH_ENABLED = os.getenv("VITE_AUTH_ENABLED", "true").lower() == "true"

# Frontend info for authentication
FRONTEND_INFO = {
    "app": "AI in Action Streamlit",
    "url": os.getenv("STREAMLIT_SERVER_BASE_URL", "http://localhost:8501")
}

def initialize_auth():
    """Initialize authentication state variables if they don't exist."""
    if "auth_token" not in st.session_state:
        st.session_state.auth_token = None
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None
    if "auth_expiry" not in st.session_state:
        st.session_state.auth_expiry = 0
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

def encode_token(data):
    """Encode data to base64 (equivalent to btoa in JavaScript)"""
    return base64.b64encode(json.dumps(data).encode()).decode()

def decode_token(token):
    """Decode base64 token (equivalent to atob in JavaScript)"""
    try:
        return json.loads(base64.b64decode(token).decode())
    except Exception as e:
        logger.error(f"Token decode error: {str(e)}")
        return None

def redirect_to_signin():
    """Redirect to authentication service signin page"""
    encoded_info = encode_token(FRONTEND_INFO)
    signin_url = f"{AUTH_URL}/signin/?v={encoded_info}"
    
    # Enhanced redirect with both meta refresh and JavaScript
    st.markdown(f"""
    <meta http-equiv="refresh" content="0;URL='{signin_url}'" />
    <script>
        window.location.href = "{signin_url}";
    </script>
    <p>If you are not redirected automatically, <a href="{signin_url}">click here</a>.</p>
    """, unsafe_allow_html=True)
    st.stop()

def parse_token_from_url():
    """Check if there's a token in the URL parameters."""
    token = st.query_params.get("t", None)
    
    if token:
        logger.info("Found token in URL parameters")
        # Remove token from URL to prevent issues on refresh
        st.query_params.clear()
        return token
    
    return None

def check_auth(token):
    """Verify if the token is valid by calling the authentication API."""
    if not token:
        return False
    
    try:
        # Decode token to get expiry
        token_data = json.loads(base64.b64decode(token))
        
        # Check if token is expired
        if time.time() * 1000 > token_data.get('expiry', 0):
            logger.info("Token expired, verifying with API")
            # Call API to check if token is still valid
            response = requests.get(
                f"{AUTH_URL}/auth/check/",
                headers={"x-token": token_data.get('token', '')},
                timeout=10
            )
            
            if not response.ok:
                logger.warning("API auth check failed")
                return False
            
            logger.info("API auth check successful")
            # Update token in session state
            st.session_state.auth_token = token
            st.session_state.auth_user = token_data.get('user', {})
            st.session_state.auth_expiry = token_data.get('expiry', 0)
            st.session_state.authenticated = True
            return True
        else:
            logger.info("Token still valid")
            # Store token info in session state
            st.session_state.auth_token = token
            st.session_state.auth_user = token_data.get('user', {})
            st.session_state.auth_expiry = token_data.get('expiry', 0)
            st.session_state.authenticated = True
            return True
    
    except Exception as e:
        logger.error(f"Auth check error: {str(e)}")
        return False

def get_username():
    """Get username from token if available"""
    if st.session_state.auth_user and 'name' in st.session_state.auth_user:
        return st.session_state.auth_user['name']
    return "User"
    
def logout():
    """Clear auth session data and redirect to sign-in"""
    logger.info("Logging out user")
    
    # Preserve auth-related keys but set to None/False
    st.session_state.auth_token = None
    st.session_state.auth_user = None
    st.session_state.auth_expiry = 0
    st.session_state.authenticated = False
    
    # Clear all non-auth session state to reset the app
    for key in list(st.session_state.keys()):
        if key not in ["auth_token", "auth_user", "auth_expiry", "authenticated"]:
            del st.session_state[key]
    
    # Force redirect to signin page after clearing session
    redirect_to_signin()

def is_authenticated():
    """Check if user is authenticated"""
    return st.session_state.get("authenticated", False)

def require_auth():
    """Main authentication function to be used in Streamlit apps"""
    if not AUTH_ENABLED:
        return True
        
    # Initialize authentication state
    initialize_auth()
    
    # Check if already authenticated in this session
    if st.session_state.authenticated:
        return True
        
    # Check for token in query params
    token_from_url = parse_token_from_url()
    
    if token_from_url:
        # Validate token
        is_valid = check_auth(token_from_url)
        if is_valid:
            st.session_state.auth_token = token_from_url
            st.session_state.authenticated = True
            return True
        else:
            redirect_to_signin()
            
    # Check for token in session state
    elif "auth_token" in st.session_state and st.session_state.auth_token:
        is_valid = check_auth(st.session_state.auth_token)
        if is_valid:
            st.session_state.authenticated = True
            return True
        else:
            st.session_state.auth_token = None
            redirect_to_signin()
    
    # No token found
    else:
        redirect_to_signin()
        
    return False
