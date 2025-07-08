import re
import os
import requests
from dotenv import load_dotenv
import json

def format_plan_to_json(raw_text, query):
    lines = raw_text.strip().split("\n")
    subtasks = []
    current_task = None

    for line in lines:
        task_match = re.match(r'^\d+\.\s+\*\*(.+?)\*\*', line)  # e.g., numbered + bold
        if not task_match:
            task_match = re.match(r'^\d+\.\s+(.+)', line)  # fallback: numbered + plain
        bullet_match = re.match(r'^\s*-\s+(.*)', line)  # bullet points
        if task_match:
            if current_task and current_task["details"]:  # only save if it had bullets
                subtasks.append(current_task)
            current_task = {"title": task_match.group(1).strip(), "details": []}
        elif bullet_match and current_task:
            current_task["details"].append(bullet_match.group(1).strip())
    if current_task and current_task["details"]:
        subtasks.append(current_task)

    return json.dumps({"query": query, "subtasks": subtasks}, indent=2)



def call_openrouter(prompt):
    load_dotenv()
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost"
    }
    payload = {
        "model": "openai/gpt-3.5-turbo-0613gi",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=payload)

    try:
        data = response.json()
    except Exception as e:
        raise Exception(f"Failed to parse response as JSON: {e}\nRaw response:\n{response.text}")

    if "choices" not in data:
        raise Exception(f"API Error: Missing 'choices' in response.\nFull response:\n{json.dumps(data, indent=2)}")

    return data["choices"][0]["message"]["content"]