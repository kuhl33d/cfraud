#!/bin/bash

# Set default database credentials if not provided
if [ -z "$1" ]; then
    DB_USER="root"
else
    DB_USER="$1"
fi

if [ -z "$2" ]; then
    DB_PASS="password"
else
    DB_PASS="$2"
fi

# Configure database connection
export DATABASE_URL="mysql+mysqlconnector://${DB_USER}:${DB_PASS}@localhost/cfraud"

echo "Database connection configured with user: ${DB_USER}"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
    echo "Virtual environment created."
fi

# Activate virtual environment and install dependencies
echo "Activating virtual environment and installing dependencies..."
source .venv/bin/activate
pip install -r requirements.txt

echo ""
echo "Starting Flask application..."
echo "Access the application at http://localhost:5000"
echo ""

# Run the Flask application
python -m flask --app backend/app run --debug