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

# Get latest commit ID
export COMMIT_SHA=$(git rev-parse --short HEAD)

# Check if the command worked
if [ -z "$COMMIT_SHA" ]; then
    echo "Error: Could not get commit SHA."
    echo "Make sure you are running this script from within a git repository."
    exit 1
fi

# Finally, deploy
gcloud builds submit --config cloudbuild.yaml --substitutions=SHORT_SHA=$COMMIT_SHA
