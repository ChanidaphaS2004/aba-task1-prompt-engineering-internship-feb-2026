# %% [1] @title 1. Library และการตั้งค่า Local Path
import os
import json
import re
import pandas as pd
import time
from openai import OpenAI
from collections import Counter
from tqdm import tqdm

# ตั้งค่าที่อยู่ไฟล์ในเครื่อง (Local Path)
INPUT_CSV_PATH = "../Data/data_sentiment_no_Off.csv" 
OUTPUT_COMBINED_PATH = "../Result/Sentiment_All_Results_0-shotCoT_no_TopicOff_1.5b.csv"

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
    
    # 1. สกัด Thinking Log (<think>...</think>)
    think_match = re.search(r'<think>(.*?)</think>', full_output, re.DOTALL)
    think_log = think_match.group(1).strip() if think_match else ""
    
    # 2. สกัด JSON ดิบ 
    clean_text_for_json = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL)
    json_match = re.search(r'\{.*\}', clean_text_for_json, re.DOTALL)
    
    raw_json_str = ""
    sentiment = "Unknown"

    if json_match:
        # ดึงก้อน JSON ดิบออกมา
        raw_json_str = json_match.group().strip()
        
        raw_json_str = raw_json_str.replace("'", '"')
        
        try:
            # Parse เพื่อเอา Sentiment มาทำ Majority Vote
            data = json.loads(raw_json_str)
            sentiment = data.get("sentiment", "Unknown").capitalize()
        except:
            match = re.search(r'\b(Positive|Negative)\b', raw_json_str, re.IGNORECASE)
            if match: sentiment = match.group(1).capitalize()
    
    # 3. รวม "สิ่งที่คิด" กับ "ก้อน JSON" เข้าด้วยกันเพื่อลงช่อง Reasoning
    final_reasoning = f"THINKING:\n{think_log}\n\nRAW_JSON:\n{raw_json_str}".strip()
    
    if not think_log and not json_match:
        final_reasoning = full_output

    return {
        "sentiment": sentiment,
        "reasoning": final_reasoning
    }

def run_inference(content):
    system_msg = (
        "You are a specialized expert in Sentiment Analysis and Natural Language Understanding."
        "Task: Classify the sentiment of the 'Selected Content' \n"
        "Sentiment Classification Rules:\n"
        "- Positive: The content indicates satisfaction, praise, or a favorable stance.\n"
        "- Negative: The content expresses dissatisfaction, complaints, or unfavorable opinions.\n"
        "Determine the overall sentiment and choose exactly one value from ['Positive', 'Negative'].\n"
        "Response Format: You must output ONLY a valid JSON object:\n"
        "{"        
        "  \"reasoning_trace\": \"A brief explanation of why this conclusion was reached\",\n"
        "  \"sentiment\": \"Positive | Negative\"\n"
        "}"
    )
    #user_msg = f"Target Topic: {topic}\nInput Content: {content}\n\nLet's think step by step."
    user_msg = f"Input Content: {content}\n\nLet's think step by step."

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
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

print(f"🚀 เริ่มรัน {MODEL_NAME} (Majority Vote 3 รอบ)...")

for idx, row in tqdm(df_input.iterrows(), total=len(df_input)):
    #topic = row['Topic']
    content = row['Selected Content']
    
    rounds = []
    for r in range(3):
        raw = run_inference(content)
        parsed = parse_deepseek_response(raw)
        rounds.append(parsed)
    
    # Majority Vote
    voted_sentiment = Counter([d['sentiment'] for d in rounds]).most_common(1)[0][0]

    # บันทึกข้อมูล
    combined_results.append({
        "ID": row['ID'],
        #"Topic": topic,
        "Selected Content": content,
        "Final_Vote": voted_sentiment,
        # Round 1
        "Round1_Reasoning": rounds[0]['reasoning'],
        "Round1_Output": rounds[0]['sentiment'],
        # Round 2
        "Round2_Reasoning": rounds[1]['reasoning'],
        "Round2_Output": rounds[1]['sentiment'],
        # Round 3
        "Round3_Reasoning": rounds[2]['reasoning'],
        "Round3_Output": rounds[2]['sentiment']
    })

    if (idx + 1) % 5 == 0:
        pd.DataFrame(combined_results).to_csv(OUTPUT_COMBINED_PATH, index=False, encoding='utf-8-sig')

pd.DataFrame(combined_results).to_csv(OUTPUT_COMBINED_PATH, index=False, encoding='utf-8-sig')
print(f"✅ บันทึกเสร็จแล้วที่: {OUTPUT_COMBINED_PATH}")
