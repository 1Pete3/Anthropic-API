from textwrap import indent
from anthropic import Anthropic
from dotenv import load_dotenv
import json
import os

load_dotenv()
client = Anthropic()
messages = []


def main():
    if not os.path.exists("G:\\Coding\\Anthropic\\API Tutorial\\dataset.json"):
        dataset = generate_dataset()
        with open("dataset.json", "w") as f:
            json.dump(dataset, f, indent=2)

    with open("dataset.json", "r") as f:
        dataset = json.load(f)

    eval_results = run_eval(dataset)
    with open("results.json", "w") as f:
        json.dump(eval_results, f, indent=2)


def generate_dataset():
    prompt = """
                Generate an evaluation dataset for a prompt evaluation. The dataset will be used to evaluate prompts that generate Python, JSON, or Regex specifically for AWS-related tasks. Generate an array of JSON objects, each representing task that requires Python, JSON, or a Regex to complete.

                Example output:
                ```json
                [
                \{
                    "task": "Description of task",
                \},
                ...additional
                ]
                ```

                * Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a single regex
                * Focus on tasks that do not require writing much code

                Please generate 3 objects.
                """
    add_user_message(messages, prompt)
    add_assistant_message(messages, "```json")
    dataset = chat(messages, stop_sequences=["```"])
    return json.loads(dataset)


# --- Helper functions
def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})


def chat(messages, system=None, temperature=1.0, stop_sequences=None):

    params = {"model": "claude-haiku-4-5", "max_tokens": 1000, "messages": messages}
    if system:
        params["system"] = system
    if stop_sequences:
        params["stop_sequences"] = stop_sequences
    output_text = client.messages.create(**params)
    return output_text.content[0].text


# --- Model evaluation
def run_prompt(test_case):
    """Merges the prompt and test case input, then returns the results"""
    prompt = f"""
    Please solve the following task:
    
    {test_case["task"]}
    """
    add_user_message(messages, prompt)
    output = chat(messages)
    return output


def run_test_case(test_case):
    output = run_prompt(test_case)

    # TODO - Grading
    score = 10

    return {"output": output, "test_case": test_case, "score": score}


def run_eval(dataset):
    results = []

    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)
    return results


# --- Main function
if __name__ == "__main__":
    main()
