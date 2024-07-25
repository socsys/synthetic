import torch
from transformers import pipeline
import pandas as pd
import os

pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")


user_msg = '''
Change the original post while  following these rules:

Change the original post while  following these rules:
1. Replace all non-sensitive private information such as age, dob, religion, gender, marital status, race, ethnicity, employment, location, sexuality, and parenthood with other non-sensitive private information that retains the context. Replace the organization name with any other organization that serves the same purpose without generalization.
2. Change specific codes, IDs, numbers, and names with different codes, IDs, numbers, and names, respectively.
3. Use Reddit-style internet language. Ensure the generated text resonates with the Reddit community.
4. Use common internet abbreviations and expressions where appropriate.
5.  Don't give the title of the post.
Original post='''

    # Define the user message
sys_msg = '''You are a story recreator, who takes the information from the original post,
and then makes another different story with similar kind of personal information. You want to minimize the chance of finding the link between the stories Generate the post following this format:
"Changed post":.
'''
#path = '/Original_Dataset/'
    #file_path = os.path.join(path, 'Llamma_test.csv')
#file_path = os.path.join(path, 'Zephyr.csv')

    # Open the CSV file
df = pd.read_csv("./original_dataset.csv")

    # Access a column (assuming 'model_output' is the column name)
input_col = df['original_post']

generated_text_col = []
for original_post in input_col:
        #print(original_post)
    user_post = user_msg+original_post
# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    messages = [
    {
        "role": "system",
        "content": sys_msg},
    
        {"role": "user", "content": user_post},
        ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        #print(outputs[0]["generated_text"])
    output=outputs[0]["generated_text"][len(prompt):]
    generated_text_col.append(output)
        # Check if 'generated_text' already exists
if 'generated_text' in df.columns:
        # Replace the existing 'generated_text' with the new values
    df['generated_text'] = generated_text_col
else:
      # Insert the new column after 'model_output'
    df.insert(df.columns.get_loc('original_post') + 1, 'generated_text', generated_text_col)

      # Save the updated dataframe to a new CSV file
df.to_csv("./Zephyr_generated.csv", index=False)    


