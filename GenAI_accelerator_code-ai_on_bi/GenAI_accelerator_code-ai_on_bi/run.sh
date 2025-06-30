#! /bin/bash

set -xueo pipefail

python populate_vector_db.py

python -m \
    streamlit run new_app.py \
    --server.port 8000 \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false
