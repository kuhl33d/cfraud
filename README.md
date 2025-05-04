# Credit Fraud Detection System

A Flask web application for credit card fraud detection with user authentication using MySQL database.

## Features

- User authentication (login/signup) with MySQL database
- Credit card transaction analysis
- Fraud detection using machine learning
- Interactive dashboard with data visualization
- Custom dataset upload and analysis

## Getting Started

### Prerequisites

- Python 3.7 or higher
- MySQL Server installed and running
- pip (Python package manager)

### Database Setup

1. Create the MySQL database and tables by running the SQL script:
   ```
   mysql -u root -p < backend/schema.sql
   ```

2. Configure the database connection by setting the environment variable or using the provided scripts.

### Installation

#### Option 1: Using the provided scripts

**For Windows:**
```
run_app.bat [mysql_username] [mysql_password]
```

**For Linux/Mac:**
```
chmod +x run_app.sh
./run_app.sh [mysql_username] [mysql_password]
```

#### Option 2: Manual setup

1. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set the database connection environment variable:
   ```
   set DATABASE_URL=mysql+mysqlconnector://username:password@localhost/cfraud  # Windows
   export DATABASE_URL=mysql+mysqlconnector://username:password@localhost/cfraud  # Linux/Mac
   ```

4. Run the application:
   ```
   python -m flask --app backend/app run --debug
   ```

5. Access the application at http://localhost:5000

Previews should run automatically when starting a workspace.