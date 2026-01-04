import json
import os

def main():
    input_file = r"datasets\twitter15\source_tweets.txt"
    output_dir = r"tweet_json_files"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    counter = 1
    for line in lines:
        # Split the line by the first tab to separate the ID and the rest
        parts = line.strip().split("\t", 1)
        if len(parts) < 2:
            continue  # Skip malformed lines

        tweet_id = parts[0]
        tweet_text = parts[1]

        # Create a dictionary for the tweet
        tweet_data = {
            "id": tweet_id,
            "text": tweet_text
        }

        # Define the output JSON file path with sequential numbering
        output_file = os.path.join(output_dir, f"tweet_id{counter}.json")

        # Write the tweet data to a JSON file
        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(tweet_data, json_file, ensure_ascii=False, indent=4)

        counter += 1

    print(f"Converted {len(lines)} tweets to JSON files in '{output_dir}'.")
    print(f"Files named: tweet_id1.json, tweet_id2.json, ..., tweet_id{len(lines)}.json")

if __name__ == "__main__":
    main()