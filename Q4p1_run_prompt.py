import openai
import os

# Set your OpenAI API key here
# openai.api_key = 

role_txt = "You are a helpful assistant to a doctor and want to predict whether the case patients die or survive. Return for each case 1-5 whether you think the patient will die or survive based on their values, and considering the examples given. Explain in one sentence in which values you based your prediction. Please know that I give more value to predicting correctly when the patient is going to die. Please always start your response with outcome: survival or outcome:death"

# Path to the directory where the prompt files are stored
# get current directory
prompt_directory = "Q4p1_txt"

# List all files in the directory
prompt_files = [f for f in os.listdir(prompt_directory) if f.endswith('.txt')]
# get length of prompt_files
len(prompt_files)

# Iterate over the files and process them one by one
#for prompt_file_name in prompt_files:

# take only the first 3 files
for prompt_file_name in prompt_files[:1]:
    prompt_file_path = os.path.join(prompt_directory, prompt_file_name)
    
    # Read the content of the file
    with open(prompt_file_path, 'r') as file:
        prompt = file.read()

    # Call OpenAI API for each file's content
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the correct chat model
        messages=[
            {"role": "system", "content": role_txt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )

    # Print the response for each file
    print(f"Response for {prompt_file_name}:")
    print(response['choices'][0]['message']['content'])
    # save the response to a file
    with open(f"Q4p1_results/{prompt_file_name}_response.txt", 'w') as response_file:
        response_file.write(response['choices'][0]['message']['content'])
    print("\n")
