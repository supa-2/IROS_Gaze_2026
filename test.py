from inference_sdk import InferenceHTTPClient
rf = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key="T2jBgkH6EwyrLDyNlSSa")
print(rf.infer("data/1.jpg", model_id="sam3/concept_segment", parameters={"prompts":[{"text":"painting"}], "format":"polygon"}))