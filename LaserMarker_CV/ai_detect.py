import os
# 1. Import the library
from inference_sdk import InferenceHTTPClient

# 2. Connect to your workflow
# Set ROBOFLOW_API_KEY in your environment or a .env file — never hardcode it here
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"]
)

# 3. Run your workflow on an image
result = client.run_workflow(
    workspace_name="franklins-workspace-eoaqd",
    workflow_id="find-tubes-and-tubes",
    images={
        "image": "black_white_in_the_dark.jpg" # Path to your image file
    },
    use_cache=True # Speeds up repeated requests
)

# 4. Get your results
print(result)


