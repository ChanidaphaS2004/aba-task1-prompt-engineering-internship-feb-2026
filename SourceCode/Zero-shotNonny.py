# %% [1] @title 1. Library และการตั้งค่า Local Path
import os
import json
import re
import pandas as pd
from openai import OpenAI
from collections import Counter
from tqdm import tqdm

# ตั้งค่าที่อยู่ไฟล์
INPUT_CSV_PATH = "./data/data_sentiment_no_Off.csv" 
OUTPUT_COMBINED_PATH = "./result/Sentiment_All_Results_0_shot_1.5b.csv"

# %% [2] @title 2. เชื่อมต่อ Ollama (Local)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama", 
)

MODEL_NAME = "deepseek-r1:1.5b" 
print(f"✅ เชื่อมต่อ Ollama (Model: {MODEL_NAME}) เรียบร้อย")

# %% [3] @title 3. ฟังก์ชันสกัดข้อมูลด้วย 
def extract_with_regex(raw_output):
    # ลบส่วน <think> ออกก่อน
    clean_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
    
    topic_found = "Null"
    sentiment_found = "Null"
    
    topics_list = ["Room", "Staff", "Location", "Food", "Price", "Facility", "Check-in", "Check-out", "Taxi-issue", "Booking-issue", "Off-topic"]
    
    for t in topics_list:
        # ดักจับ "Topic": [ { "text": "...", "label": "..." } ]
        pattern = rf'"{t}"\s*:\s*\[\s*\{{\s*"text"\s*:\s*"(?!null|test)([^"]+)"\s*,\s*"label"\s*:\s*"(?!null|test)([^"]+)"'
        
        match = re.search(pattern, clean_output, re.IGNORECASE)
        if match:
            found_text = match.group(1).strip()
            found_sentiment = match.group(2).strip()
            
            if found_text.lower() != "test":
                topic_found = t
                sentiment_found = found_sentiment.capitalize()
                break
            
    return {
        "full_output": clean_output,
        "topic": topic_found,
        "sentiment": sentiment_found
    }

def run_inference(content):
    # ใช้ Prompt จาก Dataset 2 ในงานวิจัย Nonraphan
    prompt = f"""Please output the following [text] according to the [Constraints] in the [Output Format].
    [Constraints] The output should only be in the [Output Format], and you must classify which part of the text corresponds to which Topic in the [Topics]. Additionally, determine whether each classified element is Positive or Negative. If there is no corresponding element, put Null for both ’text’ and ’label’. The most important constraint is not to include any extra characters such as newline characters, ’json’, or backticks, or any other unnecessary text outside of the [Output Format]. If there are two or more elements of the same Topic, number each so that they do not conflict when converted to JSON formatted data. However, if they have the same NegPos label, keep them in one Text as much as possible.
    [Topics] Room, Staff, Location, Food, Price, Facility, Check-in, Check-out, Taxi-issue, Booking-issue, Off-topic
    [Output Format]
    {{
    "Topics": {{ "Room": [{{ "text": "test", "label": "Positive" }}],
    "Staff": [{{ "text": null, "label": null }}],
    "Location": [{{ "text": "test", "label": "Positive" }}],
    "Food": [{{ "text": "test", "label": "Negative" }}],
    "Price": [{{ "text": "test", "label": "Positive" }}],
    "Facility": [{{ "text": "test", "label": "Negative" }}],
    "Check-in": [{{ "text": "test", "label": "Positive" }}],
    "Check-out": [{{ "text": null, "label": null }}],
    "Taxi-issue": [{{ "text": null, "label": null }}],
    "Booking-issue": [{{ "text": null, "label": null }}],
    "Off-topic": [{{ "text": null, "label": null }}]
    }} }}

[text]
{content}"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,      
            top_p=0.95,      
            max_tokens=1024     
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# %% [4] @title 4. เริ่มประมวลผล
df_input = pd.read_csv(INPUT_CSV_PATH)
combined_results = []

print(f"🚀 เริ่มรัน {MODEL_NAME} (Zero-shot Benchmarking Mode)...")

for idx, row in tqdm(df_input.iterrows(), total=len(df_input)):
    content = row['Selected Content']
    ground_truth_topic = row.get('Topic', 'Unknown')
    
    rounds = []
    for r in range(3):
        raw_response = run_inference(content)
        parsed = extract_with_regex(raw_response)
        rounds.append(parsed)
    
    voted_topic = Counter([d['topic'] for d in rounds]).most_common(1)[0][0]
    voted_sentiment = Counter([d['sentiment'] for d in rounds]).most_common(1)[0][0]

    combined_results.append({
        "ID": row['ID'],
        "Topic": ground_truth_topic,
        "Selected Content": content,
        "Final_Vote_topic": voted_topic,
        "Final_Vote_sentiment": voted_sentiment,
        "Round1_Output": rounds[0]['full_output'],
        "Round1_Output_topic": rounds[0]['topic'],
        "Round1_Output_sentiment": rounds[0]['sentiment'],
        "Round2_Output": rounds[1]['full_output'],
        "Round2_Output_topic": rounds[1]['topic'],
        "Round2_Output_sentiment": rounds[1]['sentiment'],
        "Round3_Output": rounds[2]['full_output'],
        "Round3_Output_topic": rounds[2]['topic'],
        "Round3_Output_sentiment": rounds[2]['sentiment']
    })

    if (idx + 1) % 5 == 0:
        pd.DataFrame(combined_results).to_csv(OUTPUT_COMBINED_PATH, index=False, encoding='utf-8-sig')

pd.DataFrame(combined_results).to_csv(OUTPUT_COMBINED_PATH, index=False, encoding='utf-8-sig')
print(f"✅ เสร็จแล้ว! ไฟล์ผลลัพธ์อยู่ที่: {OUTPUT_COMBINED_PATH}")
