# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import base image
FROM python:3.13-slim

# set working dir
WORKDIR /app

# copy/install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# add user: optional
RUN adduser --disabled-password --gecos "" myuser && \
    chown -R myuser:myuser /app

# copy all files from src to /app
COPY . .

# switch user
USER myuser

# set environment path
ENV PATH="/home/myuser/.local/bin:$PATH"

# Run the streamlit UI
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
