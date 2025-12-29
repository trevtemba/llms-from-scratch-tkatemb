import json
from tqdm import tqdm
import urllib.request

file_path = "instruction-data-with-response.json"
with open(file_path, "r") as file:
    test_data = json.load(file)


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text

def query_model(
    prompt,
    model="llama3",
    url="http://localhost:11434/api/chat"
):
    data= {
        "model":model,
        "messages": [
            {
                "role":"user",
                "content":prompt
            }
        ],
        "options": {
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    )

    request.add_header("Content-Type", "application/json")

    response_data = ""

    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line :
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data

# Just returns 0-100 so that we can calculate an average score for our model much more easily
def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}` "
            f"on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue
    return scores

if __name__ == "__main__":
    model = "llama3"

    ### Original method for scoring, much more verbose and harder to quantify at scale though.
    # for entry in test_data[:3]:
    #     prompt = (
    #         f"Given the input `{format_input(entry)}` "
    #         f"and correct output `{entry['output']}`, "
    #         f"score the model response `{entry['model_response']}` "
    #         f"on a scale from 0 to 100, where 100 is the best score." 
    #     )
    #     print("\nDataset response:")
    #     print(">>", entry['output'])
    #     print("\nModel response:")
    #     print(">>", entry["model_response"])
    #     print("\nScore:")
    #     print(">>", query_model(prompt))
    #     print("\n-------------------------")

    scores = generate_model_scores(test_data, "model_response")
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average scores: {sum(scores)/len(scores):.2f}\n")