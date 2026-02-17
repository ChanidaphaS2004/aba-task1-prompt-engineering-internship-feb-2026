# %% [1] @title 1. Library และการตั้งค่า Local Path
import os
import re
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

INPUT_CSV_PATH = "../Data/data_sentiment_no_Off.csv" 
OUTPUT_STAGE1_PATH = "../Result/Sentiment_All_Results_L2M_Stage1_1.5b.csv"

# %% [2] @title 2. เชื่อมต่อ Ollama (Local)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama", 
)

MODEL_NAME = "deepseek-r1:1.5b" 
print(f"✅ เชื่อมต่อ Ollama (Model: {MODEL_NAME}) สำหรับ Stage 1")

# %% [3] @title 3. ฟังก์ชันสำหรับ Stage 1: Decomposition (เวอร์ชัน JSON Output - Single Question)
import json
import re

def parse_decomposition_output(full_output):
    """
    สกัด JSON จาก Output ของ DeepSeek และดึงเฉพาะ key "sub_question"
    """
    clean_text = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL).strip()
    
    json_match = re.search(r'\{.*\}', clean_text, re.DOTALL)
    
    if json_match:
        try:
            data = json.loads(json_match.group())
            return data.get("sub_question", "")
        except Exception as e:
            print(f"⚠️ JSON Decode Error: {e}. Trying regex fallback...")
            match = re.search(r'"sub_question":\s*"(.*?)"', clean_text)
            if match: return match.group(1)
            
    return ""

def run_decomposition(content):
    system_msg = (
        "You are a specialized Sentiment Analysis expert. Your task is to analyze a hotel review "
        "and identify the specific information (sub-question) needed to determine the final sentiment."
    )
    
    user_msg = f"""Instruction: You are a specialized Sentiment Analysis expert. Your task is to analyze a hotel review and identify the specific information (sub-question) needed to determine the final sentiment.

    Example 1:
    Q: Content: "There was a problem with heating water at that time and we can understand, but eventually we couldn't manage to have a proper shower with hot water." What is the sentiment of this Selected Content between Positive or Negative?
    Decomposition: To answer the question "What is the sentiment of this Selected Content between Positive or Negative?", we need to know: "What was the guest's experience with the hotel's basic facilities (hot water) and was it resolved?"

    Example 2:
    Q: Content: "Nice location, not far from Airport and Mackenzie beach." What is the sentiment of this Selected Content between Positive or Negative?
    Decomposition: To answer the question "What is the sentiment of this Selected Content between Positive or Negative?", we need to know: "How does the guest evaluate the convenience and surroundings of the hotel's location?"

    Example 3:
    Q: Content: "The breakfast prepared by Richard was delicious and freshly prepared." What is the sentiment of this Selected Content between Positive or Negative?
    Decomposition: To answer the question "What is the sentiment of this Selected Content between Positive or Negative?", we need to know: "What are the specific details of the guest's feedback regarding the quality and preparation of the breakfast?"

    Example 4:
    Q: Content: "the building is completely refurbished, it is confortable" What is the sentiment of this Selected Content between Positive or Negative?
    Decomposition: To answer the question "What is the sentiment of this Selected Content between Positive or Negative?", we need to know: "How does the guest describe the physical condition and comfort level of the hotel building?"

    Example 5:
    Q: Content: "We had a late check in and there was no one in the hotel to give us keys and the door was locked." What is the sentiment of this Selected Content between Positive or Negative?
    Decomposition: To answer the question "What is the sentiment of this Selected Content between Positive or Negative?", we need to know: "What specific obstacles did the guest face during the check-in process and was there any staff assistance available?"

    Current Task:
    Q: Content: "{content}"
    Response Format: Output ONLY a JSON object with the key "sub_question" representing the information needed.
    Decomposition:"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.6, 
            top_p=0.95,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e: 
        return f"Error: {str(e)}"

# %% [4] @title 4. เริ่มประมวลผล Stage 1 
df_input = pd.read_csv(INPUT_CSV_PATH)
stage1_results = []

print(f"🚀 เริ่มรัน Stage 1 (Decomposition) พร้อม JSON Output (Single Sub-Question)...")

for idx, row in tqdm(df_input.iterrows(), total=len(df_input)):
    content = row['Selected Content']
    
    raw_output = run_decomposition(content)
    sub_q = parse_decomposition_output(raw_output)

    stage1_results.append({
        "ID": row['ID'],
        "Selected Content": content,
        "Sub_Question": sub_q, 
        "Raw_Stage1_Output": raw_output
    })

    if (idx + 1) % 10 == 0:
        pd.DataFrame(stage1_results).to_csv(OUTPUT_STAGE1_PATH, index=False, encoding='utf-8-sig')

# บันทึกไฟล์สุดท้าย
df_final = pd.DataFrame(stage1_results)
df_final.to_csv(OUTPUT_STAGE1_PATH, index=False, encoding='utf-8-sig')

print(f"\n✅ Stage 1 เสร็จสมบูรณ์! พร้อมนำ 'Sub_Question' ไปใช้ต่อใน Stage 2")
print(f"📁 บันทึกที่: {OUTPUT_STAGE1_PATH}")
