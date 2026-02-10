# %% [1] @title 1. Library และการตั้งค่า Local Path
import os
import json
import re
import pandas as pd
import time
from openai import OpenAI
from collections import Counter
from tqdm import tqdm

INPUT_CSV_PATH = "./data/data_sentiment_no_Off.csv" 
OUTPUT_COMBINED_PATH = "./result/Sentiment_All_ResultsABSA_noReasoning_noOff_1.5b.csv"

print("✅ โหลด Library และตั้งค่า Path เรียบร้อย")

# %% [2] @title 2. เชื่อมต่อ Ollama (Local)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama", 
)

MODEL_NAME = "deepseek-r1:1.5b" 

print(f"✅ เชื่อมต่อ Ollama (Model: {MODEL_NAME}) เรียบร้อย")

# %% [3] @title 3. ฟังก์ชันสกัดข้อมูล
def parse_deepseek_response(full_output):

    # 1. สกัดเฉพาะข้อความภายใน <think>...</think> 
    think_match = re.search(r'<think>(.*?)</think>', full_output, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else ""

    # 2. สกัดก้อน JSON {...}
    json_match = re.search(r'\{.*\}', full_output, re.DOTALL)
    json_raw = json_match.group(0).strip() if json_match else ""

    # 3. รวม "เนื้อหาการคิด" + "ก้อน JSON" เพื่อลงช่อง Reasoning
    combined_reasoning = f"{think_content}\n\n{json_raw}".strip()
    
    if not combined_reasoning:
        combined_reasoning = full_output

    # 4. สกัด Sentiment เพื่อไปลงช่อง Output และทำ Majority Vote
    sentiment = "Unknown"
    if json_match:
        try:
            data = json.loads(json_raw.replace("'", '"')) 
            sentiment = data.get("sentiment_polarity", "Unknown").capitalize()
        except:
            pass
            
    if sentiment == "Unknown":
        clean_text = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL)
        match = re.search(r'\b(Positive|Negative)\b', clean_text, re.IGNORECASE)
        if match: sentiment = match.group(1).capitalize()

    return {
        "sentiment": sentiment,
        "reasoning": combined_reasoning
    }

def run_inference(topic, content):
    system_msg = (
        "You are a specialized expert in Aspect-Based Sentiment Analysis (ABSA). "
        "ABSA requires identifying fine-grained aspect terms and their corresponding sentiment polarities "
        "by evaluating specific opinion expressions within the text.\n\n"
        "Task: Classify the sentiment polarity of the 'Selected Content' "
        "strictly in relation to the provided 'Topic' (Aspect Category).\n\n"
        "Valid Topics: Booking-issue, Check-in, Check-out, Facility, Food, "
        "Location, Price, Room, Staff, Taxi-issue.\n\n"
        "Sentiment Polarity Classification Rules:\n"
        "- Positive: The content contains explicit or implicit opinion terms indicating "
        "satisfaction, praise, or a favorable stance toward the specific aspect[cite: 15, 16].\n"
        "- Negative: The content expresses dissatisfaction, complaints, or unfavorable opinions "
        "regarding the specified aspect[cite: 15, 20].\n\n"
        "Analysis Requirement: Explicitly model the relationship between the aspect and the "
        "potential opinion terms before making a judgment.\n\n"
        "Response Format: You must output ONLY a valid JSON object:\n"
        "{\n"
        "  \"sentiment_polarity\": \"Positive | Negative\",\n"
        "  \"opinion_term\": \"The specific words indicating the sentiment\""
        "}"
    )
    user_msg = f"Target Topic: {topic}\nInput Content: {content}\n\nLet's think step by step."

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.6,
            top_p=0.95,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# %% [4] @title 4. เริ่มประมวลผล
df_input = pd.read_csv(INPUT_CSV_PATH) 
combined_results = []

print(f"🚀 เริ่มรัน {MODEL_NAME} (Majority Vote 3 รอบ)...")

for idx, row in tqdm(df_input.iterrows(), total=len(df_input)):
    topic = row['Topic']
    content = row['Selected Content']
    
    rounds_data = []

    for r in range(3):
        raw_output = run_inference(topic, content)
        parsed = parse_deepseek_response(raw_output)
        rounds_data.append(parsed)
    
    # 1. คำนวณ Majority Vote 
    sentiments_only = [rd['sentiment'] for rd in rounds_data]
    voted_sentiment = Counter(sentiments_only).most_common(1)[0][0]

    # 2. บันทึกข้อมูล
    combined_results.append({
        "ID": row['ID'],
        "Topic": topic,
        "Selected Content": content,
        "Final_Vote": voted_sentiment,
        
        # รอบที่ 1: Reasoning 
        "Round1_Reasoning": rounds_data[0]['reasoning'],
        "Round1_Output": rounds_data[0]['sentiment'],
        
        # รอบที่ 2
        "Round2_Reasoning": rounds_data[1]['reasoning'],
        "Round2_Output": rounds_data[1]['sentiment'],
        
        # รอบที่ 3
        "Round3_Reasoning": rounds_data[2]['reasoning'],
        "Round3_Output": rounds_data[2]['sentiment']
    })
    # Auto-save ทุกๆ 5 แถว 
    if (idx + 1) % 5 == 0:
        pd.DataFrame(combined_results).to_csv(OUTPUT_COMBINED_PATH, index=False, encoding='utf-8-sig')

pd.DataFrame(combined_results).to_csv(OUTPUT_COMBINED_PATH, index=False, encoding='utf-8-sig')
print(f"✅ บันทึกเสร็จสมบูรณ์: {OUTPUT_COMBINED_PATH}")