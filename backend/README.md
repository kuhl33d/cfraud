# Credit Fraud Detection System - MySQL Setup

This document provides instructions for setting up the MySQL database for the Credit Fraud Detection application.

## Database Setup

### Prerequisites

- MySQL Server installed and running
- MySQL client or MySQL Workbench for executing SQL commands

### Setup Steps

1. **Create the Database and Tables**

   You can create the database and tables by executing the SQL commands in the `schema.sql` file. There are two ways to do this:

   **Option 1: Using MySQL command line client:**
   ```bash
   mysql -u root -p < schema.sql
   ```

   **Option 2: Using MySQL Workbench:**
   - Open MySQL Workbench
   - Connect to your MySQL server
   - Go to File > Open SQL Script
   - Select the schema.sql file
   - Execute the script (lightning bolt icon)

2. **Configure Database Connection**

   The application uses environment variables to connect to the database. You need to set the following environment variable:

   ```
   DATABASE_URL=mysql+mysqlconnector://username:password@localhost/cfraud
   ```

   Replace `username` and `password` with your MySQL credentials.

   **For Windows:**
   ```cmd
   set DATABASE_URL=mysql+mysqlconnector://username:password@localhost/cfraud
   ```

   **For Linux/Mac:**
   ```bash
   export DATABASE_URL=mysql+mysqlconnector://username:password@localhost/cfraud
   ```

## Running the Application

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the Flask application:
   ```bash
   python -m flask --app backend/app run --debug
   ```

3. Access the application at http://localhost:5000

## Authentication Features

The application now includes user authentication with the following features:

- User registration (signup)
- User login
- Password hashing for security
- Session management
- Protected routes that require authentication

## Database Schema

The database includes the following tables:

1. **users** - Stores user account information
   - id: Primary key
   - username: Unique username
   - email: Unique email address
   - password_hash: Securely hashed password
   - created_at: Timestamp of account creation

2. **user_activity** - Tracks user activity (for future use)
   - id: Primary key
   - user_id: Foreign key to users table
   - activity_type: Type of activity
   - activity_details: Additional details
   - ip_address: User's IP address
   - created_at: Timestamp of activity