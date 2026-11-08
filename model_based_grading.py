from typing import Any
from dotenv import load_dotenv
from anthropic import Anthropic
import json
import os

load_dotenv()

client: Anthropic = Anthropic()
messages = []

def main():
    fp_dataset = f"G:\\Coding\\Anthropic\\API Tutorial\\dataset.json"


    if os.path.exists(fp_dataset):
        with open(fp_dataset, "r") as f:
            dataset = json.load(f)

    else:
        dataset = generate_dataset()
        with open("dataset.json", "w") as f:
            json.dump(dataset, f, indent=2)

    results = run_eval(dataset)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


def generate_dataset():
    
    prompt = """
                Generate an evaluation dataset for a prompt evaluation. The dataset will be used to evaluate prompts that generate Python, JSON, or Regex specifically for AWS-related tasks. Generate an array of JSON objects, each representing task that requires Python, JSON, or a Regex to complete.

                Example output:
                ```json
                [
                {
                    "task": "Description of task",
                },
                ...additional
                ]
                ```

                * Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a single regex
                * Focus on tasks that do not require writing much code

                Please generate 3 objects.
                """
    messages = []
    add_user_message(messages, prompt)
    add_assistant_message(messages, "```json")
    text = chat(messages, stop_sequences=["```"])
    return json.loads(text)


def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})


def chat(messages, system=None, stop_sequences=[]):
    params = {
        "model": "claude-haiku-4-5",
        "max_tokens": 1000,
        "messages": messages,
    }
    if system:
        params["system"] = system
    if stop_sequences:
        params["stop_sequences"] = stop_sequences

    response = client.messages.create(**params)
    return response.content[0].text


def run_prompt(test_case):
    """Merges the prompt and test case input, then returns the result"""
    prompt = f"""
Please solve the following task:

{test_case["task"]}
"""
    
    messages = []
    add_user_message(messages, prompt)
    output = chat(messages)
    return output


def run_test_case(test_case):
    """Calls run_prompt, then grades the result"""
    output = run_prompt(test_case)
    
    # TODO - Grading
    model_grade = grade_by_model(test_case,output)
    score = model_grade["score"]
    reasoning = model_grade["reasoning"]
    
    return {
        "output": output,
        "test_case": test_case,
        "score": score,
        "reasoning": reasoning
    }


def run_eval(dataset):
    """Loads the dataset and calls run_test_case with each case"""
    results = []
    
    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)
    
    return results


# --- Model Grader
def grade_by_model(test_case, output):
    # Create evaluation prompt
    eval_prompt = """
    You are an expert code reviewer. Evaluate this AI-generated solution.
    
    Task: {task}
    Solution: {solution}
    
    Provide your evaluation as a structured JSON object with:
    - "strengths": An array of 1-3 key strengths
    - "weaknesses": An array of 1-3 key areas for improvement  
    - "reasoning": A concise explanation of your assessment
    - "score": A number between 1-10

    Respond with ONLY the JSON object, using exactly these keys.
    """.format(task=test_case["task"], solution=output)

    messages: list[Any] = []
    add_user_message(messages, eval_prompt)
    add_assistant_message(messages, "```code")

    eval_text = chat(messages, stop_sequences=["```"])
    return json.loads(eval_text)


if __name__ == "__main__":
    main()
