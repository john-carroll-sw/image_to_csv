#!/bin/bash

# Example usage: ./tag_resources.sh app-demos-rg

RESOURCE_GROUP=$1

if [ -z "$RESOURCE_GROUP" ]; then
    echo "Usage: $0 <resource_group_name>"
    exit 1
fi

# Add tags to all resources in the resource group
az resource tag --tags Project=LiveDemos Environment=Demo Owner=JohnCarroll Department=Innovation \
    --resource-group $RESOURCE_GROUP

echo "Tags applied to all resources in $RESOURCE_GROUP"
