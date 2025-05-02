#!/bin/sh
source .venv/bin/activate
python -m flask --app backend/app run -p $PORT --debug