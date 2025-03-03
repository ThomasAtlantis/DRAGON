import requests
import argparse
from rich import print
import json

args = argparse.ArgumentParser()
args.add_argument("--query", '-q', type=str, required=True, help="Query to evaluate")
args.add_argument("--max_new_tokens", '-n', type=int, default=10, help="Maximum number of tokens to generate")
args.add_argument("--mode", '-m', type=str, default="tok", help="Mode of generation (tok or seq)")
args.add_argument("--shutdown", '-s', action="store_true", help="Shutdown the server")

if __name__ == "__main__":
    args = args.parse_args()
    if args.shutdown:
        url = "http://localhost:8000/quit"
        response = requests.post(url)
    else:    
        url = "http://localhost:8000/eval"
        data = {
            "query": args.query,
            "max_new_tokens": args.max_new_tokens,
            "mode": args.mode
        }
        response = requests.post(url, json=data)
        if response.status_code == 200:
            data['output'] = response.json()['output']
            print(json.dumps(data, indent=4))