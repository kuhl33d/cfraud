import os
from backend.app import app
# from flask import Flask, send_file

# app = Flask(__name__)

# @app.route("/")
# def index():
#     return send_file('src/index.html')



def main():
    app.run(port=int(os.environ.get('PORT', 5000)))

if __name__ == "__main__":
    main()
