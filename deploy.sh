#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Check if the app name is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <app_name> [entry_file]"
    echo "Example: $0 image2csv app.py"
    exit 1
fi

# Variables
APP_NAME=$1
ENTRY_FILE=${2:-"app.py"}  # Default to app.py if not provided

# Common resources
RESOURCE_GROUP="john-live-demos"
LOCATION="eastus"
ACR_NAME="jccdemo10acr"
APP_SERVICE_PLAN="jcc-demos-asp"
KEY_VAULT="jcc-demos-kv"
LOG_ANALYTICS="jcc-demos-law"

# App-specific resources
WEB_APP_NAME="jcc-demo-${APP_NAME}"
IMAGE_NAME="${APP_NAME}"
DOCKER_IMAGE_TAG="latest"

echo "Starting deployment for app: $APP_NAME"
echo "RESOURCE_GROUP: $RESOURCE_GROUP"
echo "ACR_NAME: $ACR_NAME"
echo "APP_SERVICE_PLAN: $APP_SERVICE_PLAN"
echo "WEB_APP_NAME: $WEB_APP_NAME"
echo "IMAGE_NAME: $IMAGE_NAME"
echo "ENTRY_FILE: $ENTRY_FILE"

# Check if already logged in to Azure
if ! az account show &>/dev/null; then
    echo "Logging in to Azure..."
    az login
else
    echo "Already logged in to Azure."
fi

# Set the subscription
SUBSCRIPTION_ID="$(az account show --query 'id' -o tsv)"
echo "Using subscription ID: $SUBSCRIPTION_ID"
az account set --subscription "$SUBSCRIPTION_ID"

# Create resource group if it doesn't exist
if ! az group show --name $RESOURCE_GROUP &>/dev/null; then
    echo "Creating resource group: $RESOURCE_GROUP in $LOCATION"
    az group create --name $RESOURCE_GROUP --location $LOCATION
else
    echo "Resource group $RESOURCE_GROUP already exists."
fi

# Create Azure Container Registry if it doesn't exist
if ! az acr show --name $ACR_NAME &>/dev/null; then
    echo "Creating Azure Container Registry: $ACR_NAME"
    az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true
else
    echo "Azure Container Registry $ACR_NAME already exists."
fi

# Create Key Vault if it doesn't exist
if ! az keyvault show --name $KEY_VAULT &>/dev/null; then
    echo "Creating Key Vault: $KEY_VAULT"
    az keyvault create --name $KEY_VAULT --resource-group $RESOURCE_GROUP --location $LOCATION
else
    echo "Key Vault $KEY_VAULT already exists."
fi

# Create Log Analytics workspace if it doesn't exist
if ! az monitor log-analytics workspace show --workspace-name $LOG_ANALYTICS --resource-group $RESOURCE_GROUP &>/dev/null; then
    echo "Creating Log Analytics workspace: $LOG_ANALYTICS"
    az monitor log-analytics workspace create --resource-group $RESOURCE_GROUP --workspace-name $LOG_ANALYTICS
else
    echo "Log Analytics workspace $LOG_ANALYTICS already exists."
fi

# Create App Service Plan if it doesn't exist
if ! az appservice plan show --name $APP_SERVICE_PLAN --resource-group $RESOURCE_GROUP &>/dev/null; then
    echo "Creating App Service plan: $APP_SERVICE_PLAN"
    az appservice plan create --name $APP_SERVICE_PLAN --resource-group $RESOURCE_GROUP --sku P1V3 --is-linux
else
    echo "App Service plan $APP_SERVICE_PLAN already exists."
fi

# Login to the Azure Container Registry
echo "Logging in to Azure Container Registry: $ACR_NAME"
az acr login --name $ACR_NAME

# Build the Docker image
echo "Building Docker image: $ACR_NAME.azurecr.io/$IMAGE_NAME:$DOCKER_IMAGE_TAG"
docker build --platform linux/amd64 --build-arg ENTRY_FILE=$ENTRY_FILE -t $ACR_NAME.azurecr.io/$IMAGE_NAME:$DOCKER_IMAGE_TAG .

# Push the Docker image to the Azure Container Registry
echo "Pushing Docker image to Azure Container Registry: $ACR_NAME.azurecr.io/$IMAGE_NAME:$DOCKER_IMAGE_TAG"
docker push $ACR_NAME.azurecr.io/$IMAGE_NAME:$DOCKER_IMAGE_TAG

# Create a Web App for Containers if it doesn't exist
if ! az webapp show --name $WEB_APP_NAME --resource-group $RESOURCE_GROUP &>/dev/null; then
    echo "Creating Web App for Containers: $WEB_APP_NAME"
    az webapp create --resource-group $RESOURCE_GROUP --plan $APP_SERVICE_PLAN --name $WEB_APP_NAME \
        --deployment-container-image-name $ACR_NAME.azurecr.io/$IMAGE_NAME:$DOCKER_IMAGE_TAG
else
    echo "Web App $WEB_APP_NAME already exists. Updating container image."
    az webapp config container set --name $WEB_APP_NAME --resource-group $RESOURCE_GROUP \
        --docker-custom-image-name $ACR_NAME.azurecr.io/$IMAGE_NAME:$DOCKER_IMAGE_TAG \
        --docker-registry-server-url https://$ACR_NAME.azurecr.io
fi

# Load environment variables from .env file
echo "Loading environment variables from .env file"
set -a
source .env
set +a

# Get all environment variable names from .env file
ENV_VARS=$(grep -v '^#' .env | grep '=' | cut -d '=' -f1)

# Build the appsettings command
APPSETTINGS_CMD="az webapp config appsettings set --resource-group $RESOURCE_GROUP --name $WEB_APP_NAME --settings"

# Add each environment variable to the command
for var in $ENV_VARS; do
    # Use indirect reference to get the value
    value=${!var}
    APPSETTINGS_CMD="$APPSETTINGS_CMD $var='$value'"
done

# Set environment variables in Azure Web App
echo "Setting environment variables in Azure Web App: $WEB_APP_NAME"
eval $APPSETTINGS_CMD

# Enable application logs
echo "Enabling application logs for Web App: $WEB_APP_NAME"
az webapp log config --name $WEB_APP_NAME --resource-group $RESOURCE_GROUP --application-logging true --level information

# Set up Application Insights
echo "Setting up Application Insights for Web App: $WEB_APP_NAME"
APP_INSIGHTS_NAME="${WEB_APP_NAME}-ai"

# Create Application Insights if it doesn't exist
if ! az monitor app-insights component show --app $APP_INSIGHTS_NAME --resource-group $RESOURCE_GROUP &>/dev/null; then
    echo "Creating Application Insights: $APP_INSIGHTS_NAME"
    az monitor app-insights component create --app $APP_INSIGHTS_NAME \
        --resource-group $RESOURCE_GROUP \
        --location $LOCATION \
        --kind web \
        --application-type web \
        --workspace $LOG_ANALYTICS
else
    echo "Application Insights $APP_INSIGHTS_NAME already exists."
fi

# Get the instrumentation key
INSTRUMENTATION_KEY=$(az monitor app-insights component show \
    --app $APP_INSIGHTS_NAME \
    --resource-group $RESOURCE_GROUP \
    --query instrumentationKey \
    --output tsv)

# Add Application Insights to the Web App
az webapp config appsettings set --name $WEB_APP_NAME --resource-group $RESOURCE_GROUP \
    --settings APPINSIGHTS_INSTRUMENTATIONKEY=$INSTRUMENTATION_KEY \
    APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=$INSTRUMENTATION_KEY

echo "Deployment completed. You can access your app at https://$WEB_APP_NAME.azurewebsites.net"
echo "Application Insights dashboard: https://portal.azure.com/#resource/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/microsoft.insights/components/$APP_INSIGHTS_NAME/overview"
