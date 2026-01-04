import os
import json

def test_dataset_loading():
    dataset_path = "datasets/twitter15"
    
    print("Checking dataset structure...")
    if os.path.exists(dataset_path):
        for root, dirs, files in os.walk(dataset_path):
            print(f"Directory: {root}")
            for file in files:
                print(f"  File: {file}")
    
    # Test loading a specific file
    test_file = "datasets/twitter15/source_tweets/731166399389962242.json"
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            content = json.load(f)
            print(f"Content of test file: {content}")

if __name__ == "__main__":
    test_dataset_loading()