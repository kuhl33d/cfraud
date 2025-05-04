@echo off
echo Setting up Credit Fraud Detection Application
echo ======================================

REM Set default database credentials if not provided
IF "%1"=="" (
    SET DB_USER=root
) ELSE (
    SET DB_USER=%1
)

IF "%2"=="" (
    SET DB_PASS=password
) ELSE (
    SET DB_PASS=%2
)

REM Configure database connection
SET DATABASE_URL=mysql+mysqlconnector://%DB_USER%:%DB_PASS%@localhost/cfraud

echo Database connection configured with user: %DB_USER%
echo.

REM Check if virtual environment exists
IF NOT EXIST .venv (
    echo Creating virtual environment...
    python -m venv .venv
    echo Virtual environment created.
)

REM Activate virtual environment and install dependencies
echo Activating virtual environment and installing dependencies...
CALL .venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo Starting Flask application...
echo Access the application at http://localhost:5000
echo.

REM Run the Flask application
python -m flask --app backend/app run --debug

pause