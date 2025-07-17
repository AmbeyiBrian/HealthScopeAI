#!/bin/bash
# Azure Container Apps Log Checker
# Run this script to check the actual container logs

echo "🔍 Checking Azure Container Apps Logs..."
echo "=========================================="

# First, let's check if Azure CLI is installed and logged in
echo "📋 Checking Azure CLI status..."
az account show --output table

echo ""
echo "🐳 Fetching container logs..."
echo "This will show the actual container startup and runtime logs"

# Replace these with your actual values:
RESOURCE_GROUP="rg-healthscopeai"  # Your resource group name
CONTAINER_APP_NAME="health-monitoring-app"  # Your container app name

# Get recent logs
echo "📊 Recent application logs:"
az containerapp logs show \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --follow false \
  --tail 50

echo ""
echo "🚀 Checking container revisions:"
az containerapp revision list \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --output table

echo ""
echo "⚙️ Container app details:"
az containerapp show \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "properties.configuration.ingress"
