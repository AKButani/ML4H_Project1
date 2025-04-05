import openai
import os
import time
from openai.error import RateLimitError

role_txt = "You are a helpful assistant to a doctor and want to predict whether the case patients die or survive. Return for the last case whether you think the patient will die or survive based on their values, considering the examples given, and that the death rate for 4000 cases is only 14%.State your response only and exactly with the string 'outcome:survival Index:'  or 'outcome:death  Index:' plus the original index number"

# Path to the directory where the prompt files are stored
prompt_directory = "Q4p1_txt"

# List all files in the directory
prompt_files = [f for f in os.listdir(prompt_directory) if f.endswith('.txt')]
# Get length of prompt_files
print(f"Total files: {len(prompt_files)}")

def call_openai_api_with_retry(role_txt, prompt):
    while True:
        try:
            # Make the OpenAI API call
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Correct chat model
                messages=[
                    {"role": "system", "content": role_txt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            return response  # If successful, return the response
        except RateLimitError as e:
            # Extract the wait time from the error message (Retry-After header)
            wait_time = e.headers.get('Retry-After', 60)  # Default to 60 seconds if not provided
            print(f"Rate limit reached. Retrying in {wait_time} seconds...")
            time.sleep(int(wait_time))  # Wait for the specified time before retrying

# Iterate over the files and process them one by one (adjusting to the file range you specified)
for prompt_file_name in prompt_files[67:200]:
    prompt_file_path = os.path.join(prompt_directory, prompt_file_name)
    
    # Read the content of the file
    with open(prompt_file_path, 'r') as file:
        prompt = file.read()

    # Call OpenAI API for each file's content with retry mechanism
    response = call_openai_api_with_retry(role_txt, prompt)

    # Here you can process the response as needed
    print(f"Response for file {prompt_file_name}: {response['choices'][0]['message']['content'][:200]}...")  # Print first 200 characters

     # save the response to a file
    with open(f"Q4p1_results/{prompt_file_name}_response.txt", 'w') as response_file:
        response_file.write(response['choices'][0]['message']['content'])
    print("\n")