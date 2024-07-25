# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional
import fire
from llama import Llama, Dialog
import pandas as pd
import os


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    #max_new_token=512
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Define the original post
    user_msg = '''
Change the original post while  following these rules:
1. Replace all non-sensitive private information such as age, dob, religion, gender, marital status, race, ethnicity, employment, location, sexuality, and parenthood with other non-sensitive private information that retains the context. Replace the organization name with any other organization that serves the same purpose without generalization.
2. Change specific codes, IDs, numbers, and names with different codes, IDs, numbers, and names, respectively.
3. Use Reddit-style internet language. Ensure the generated text resonates with the Reddit community.
4. Use common internet abbreviations and expressions where appropriate.
5.  Don't give the title of the post.
Original post='''

sys_msg = '''You are a story recreator who takes the information from the original post,
and then makes another different story with similar kind of personal information. You want to minimize the chance of finding the link between the stories. Generate the post following this format:
"Changed Post":
'''
path = './llama-main'
#file_path = os.path.join(path, 'post.csv')

    # Open the CSV file
df = pd.read_csv("./original_dataset.csv")

    # Access a column (assuming 'model_output' is the column name)
input_col = df['original_post']

generated_text_col = []
for original_post in input_col:
        user_post = user_msg+original_post
        #You directly prints the story rather than saying here is.... and you dont print the not about the changes made  #no benefit
        dialogs: List[Dialog] = [
            [
            {"role": "system", "content": sys_msg},
            #{"role": "assistant", "content": original_post},  # Include the original post as part of the conversation
            {"role": "user", "content": user_post}
        ]
                                ]
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p, 
        )
        for dialog, result in zip(dialogs, results):
            text = f'"{result["generation"]["content"]}"'
            index_colon = text.index(':')
            # Find the index of the occurrence of "Note" after the first colon
            #index_note = text.index('(Note')
            # Extract the substring between the first colon and the occurrence of "Note"
            substring = text[index_colon+1:].strip()
            generated_text_col.append(substring)

    # Check if 'generated_text' already exists
if 'generated_text' in df.columns:
        # Replace the existing 'generated_text' with the new values
        df['generated_text'] = generated_text_col
else:
      # Insert the new column after 'model_output'
      df.insert(df.columns.get_loc('original_post') + 1, 'generated_text', generated_text_col)

    # Save the updated dataframe to a new CSV file
df.to_csv("./Llama2generated.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)
