
# %% [1] การตั้งค่า Path และโมเดล
import os
import re
import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm
INPUT_STAGE1_PATH = "Sentiment_All_Results_L2M_Stage1_Cleaned.csv" 
OUTPUT_STAGE2_PATH = "./result/Sentiment_All_Results_L2M_Stage2_1.5b.csv"

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama", 
)
MODEL_NAME = "deepseek-r1:1.5b"

# %% [2] Cleaning & Parsing
def clean_sub_q(text):
    """ทำความสะอาด Sub_Question ที่อาจมีภาษาจีนหรือ JSON หลุดมา"""
    if pd.isna(text) or text == "": return "What is the guest's experience?"
    # ลบภาษาจีน
    text = re.sub(r'[\u4e00-\u9fff]+', '', str(text))
    # ถ้าเป็น JSON string พยายามดึงค่าออกมา
    if '"sub_question":' in text:
        match = re.search(r'"sub_question":\s*"(.*?)"', text)
        if match: text = match.group(1)
    return text.strip()

def parse_sentiment(text):
    """สกัดคำว่า Positive หรือ Negative ออกจากคำตอบสุดท้าย"""
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if re.search(r'\bPositive\b', clean, re.I): return "Positive"
    if re.search(r'\bNegative\b', clean, re.I): return "Negative"
    return "Unknown"

# %% [3] ฟังก์ชันหลักสำหรับ Stage 2 
def run_stage2_sequential(content, sub_q):
    # --- Step 2.1: หาคำตอบของปัญหาย่อย  ---
    prompt_2_1 = f"""Review Content: "{content}"
    Question: {sub_q}
    Answer the question briefly based on the review:"""
    
    try:
        res1 = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_2_1}],
            temperature=0.1
        ).choices[0].message.content
        # สกัดเอาเฉพาะเนื้อหาหลัง <think>
        ans1 = re.sub(r'<think>.*?</think>', '', res1, flags=re.DOTALL).strip()
    except Exception as e:
        ans1 = f"Error in Step 2.1: {str(e)}"

    # --- Step 2.2: สรุปผลลัพธ์สุดท้ายโดยใช้ Answer 1 (Final Decision) ---
    prompt_2_2 = f"""Review Content: "{content}"
    Analysis: {ans1}
    Final Question: What is the sentiment of this Selected Content between Positive or Negative?
    Answer (Positive or Negative only):"""

    try:
        res2 = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_2_2}],
            temperature=0.6
        ).choices[0].message.content
        final_output = re.sub(r'<think>.*?</think>', '', res2, flags=re.DOTALL).strip()
    except Exception as e:
        final_output = f"Error in Step 2.2: {str(e)}"

    return ans1, final_output

# %% [4] เริ่มกระบวนการรัน Loop
df = pd.read_csv(INPUT_STAGE1_PATH)
results = []

print(f"🚀 เริ่มรัน Stage 2 (Sequential Solving) ทั้งหมด {len(df)} แถว...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    content = row['Selected Content']
    sub_q = clean_sub_q(row['Sub_Question_Cleaned'])
    
    # รัน 2 Query ต่อเนื่อง
    answer_1, raw_final = run_stage2_sequential(content, sub_q)
    
    # สกัดขั้วอารมณ์
    final_sentiment = parse_sentiment(raw_final)
    
    results.append({
        "ID": row['ID'],
        "Content": content,
        "Sub_Question": sub_q,
        "Step_2_1_Answer": answer_1, 
        "Step_2_2_Raw": raw_final,    
        "Final_Sentiment": final_sentiment
    })

    if (idx + 1) % 10 == 0:
        pd.DataFrame(results).to_csv(OUTPUT_STAGE2_PATH, index=False, encoding='utf-8-sig')

pd.DataFrame(results).to_csv(OUTPUT_STAGE2_PATH, index=False, encoding='utf-8-sig')
print(f"\n✅ Stage 2 เสร็จสมบูรณ์! ผลลัพธ์อยู่ที่: {OUTPUT_STAGE2_PATH}")

