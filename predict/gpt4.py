# If your environment don't exist the openai library, please pip install openai
import json
import time

import openai


def gpt4(
    input_file,
    output_file,
    openai_api_key,
):
    # read dataset as dict
    with open(input_file, "r") as fi:
        df_list_in_dict = json.load(fi)
    # This key is used only for this file. Pleaes do not use or distribute this api_key outside.
    openai.api_key = openai_api_key

    # save the results of inference using GhatGPT, including Prompt, to a file
    with open("output_file_using_gpt4_log.txt", "w") as fo:
        cnt = 0
        for id in range(len(df_list_in_dict)):
            df_list_in_dict[id]["id"] = id
            print(f"id: {id}")
            # create a prompt
            prompt = f""
            print(prompt)
            print("-" * 10)
            # inference using GPT-4
            try:
                df_list_in_dict[id]["prompt"] = prompt
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "user", "content": f"{prompt}"},
                    ],
                )
                # add GPT-4 inferences
                df_list_in_dict[id]["inference"] = response["choices"][0]["message"]["content"]
                print(f'(inference): {df_list_in_dict[id]["inference"]}')
                print("\n", "=" * 50, "\n")
                # save all print output to output_file_using_gpt4_log.txt
                fo.write(
                    f'id: {id}\n{prompt}\n{"-"*10}\n(inference): {response["choices"][0]["message"]["content"]}\n\n'
                )
                time.sleep(1)
                cnt += 1
                if cnt % 100 == 0:
                    time.sleep(10)
            except Exception as e:
                fo.write(f"{e}")
                print(e)
                time.sleep(10)
    # save df_dict as json
    with open(output_file, "w") as fo:
        json.dump(df_list_in_dict, fo)
