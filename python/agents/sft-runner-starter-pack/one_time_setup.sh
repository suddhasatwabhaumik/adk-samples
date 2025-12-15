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

# global values
export PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')
export PROJECT_ID=$(gcloud projects describe $(gcloud config get-value project) --format='value(projectId)')
export COMPUTE_SA=$PROJECT_NUMBER"-compute@developer.gserviceaccount.com"

# Set project and auth
gcloud auth login
gcloud auth application-default login
export GOOGLE_APPLICATION_CREDENTIALS="~/.config/gcloud/application_default_credentials.json"
gcloud config set project $PROJECT_ID

# Enable required services
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com

# Create arfifact registry repository
gcloud artifacts repositories create sft-runner-starter-pack \
    --repository-format=docker \
    --location=us-central1 \
    --description="Docker repository for SFT Runner Starter Pack"

# Grant Permissions to the Compute Engine SA
echo "Granting roles to $COMPUTE_SA..."

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$COMPUTE_SA" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$COMPUTE_SA" \
    --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$COMPUTE_SA" \
    --role="roles/iam.serviceAccountUser"
