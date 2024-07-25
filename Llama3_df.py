#for commandline
#First login in your hugging face account
 
#pass acess token 

#For Notebooks 
#from huggingface_hub import notebook_login
#notebook_login()
import os
import torch
import pandas as pd

#print("PyTorch version:", torch.__version__)
#print("CUDA available:", torch.cuda.is_available())
#print("CUDA version:", torch.version.cuda)
#print("Number of GPUs:", torch.cuda.device_count())
#print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")



import transformers
import torch
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
"text-generation",
model=model_id,
model_kwargs={"torch_dtype": torch.bfloat16},
device="cuda",
)

user_msg = '''
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





    # Open the CSV file
df = pd.read_csv("./original_dataset.csv")

    # Access a column (assuming 'model_output' is the column name)
input_col = df['original_post']

generated_text_col = []
for original_post in input_col:
        #print(original_post)
    user_post = user_msg+original_post


    messages = [
    {"role": "system", "content": sys_msg},
    {"role": "user", "content": user_post},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    )

    terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    )
#print(outputs[0]["generated_text"][len(prompt):])
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
df.to_csv("./Llama3_generated.csv", index=False)    
