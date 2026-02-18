#Code สำหรับPromt: Least to Most แบบสะสมบริบท
# ---------------------------------------------------
# %% [1] การตั้งค่าเบื้องต้น
import os
import re
import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm

INPUT_CSV_PATH = "../Data/data_sentiment_no_Off.csv" 
OUTPUT_PATH = "../Result/Sentiment_All_Results_LtoM_1.5b.csv"

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL_NAME = "deepseek-r1:1.5b"

# %% [2] ฟังก์ชันช่วยจัดการ Text
def clean_ai_output(text):
    clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if "Decomposition:" in clean_text:
        clean_text = clean_text.split("Decomposition:")[-1].strip()
    return clean_text

def extract_sub_q(raw_output):
    """สกัดคำถามย่อยแบบ Plain Text - เวอร์ชันเข้มงวดห้ามช่วย AI"""
    sub_q = clean_ai_output(raw_output)
    return sub_q

def parse_label(text):
    """สกัด Positive/Negative จากคำตอบสุดท้าย"""
    if re.search(r'\bPositive\b', text, re.I): return "Positive"
    if re.search(r'\bNegative\b', text, re.I): return "Negative"
    return "Unknown"

# %% [3] ฟังก์ชัน Sequential Pipeline (Stage 1 -> 2.1 -> 2.2)
def run_lto_m_expert_sequential(content):
    # --- รอบที่ 1: Stage 1 - Decomposition  ---
    system_msg = "You are a specialized Sentiment Analysis expert."
    user_msg_1 = f"""Instruction: You are a specialized Sentiment Analysis expert. Your task is to analyze a hotel review and identify the specific sub-question needed to determine the final sentiment.

    Example 1: 
    Q: Content: "There was a problem with heating water at that time and we can understand, but eventually we couldn't manage to have a proper shower with hot water." What is the sentiment of this Selected Content between Positive or Negative?
    Decomposition: To answer the question "What is the sentiment of this Selected Content between Positive or Negative?", we need to know: "What was the guest's experience with the hotel's basic facilities (hot water) and was it resolved?"

    Example 2: 
    Q: Content: "Nice location, not far from Airport and Mackenzie beach." What is the sentiment of this Selected Content between Positive or Negative?
    Decomposition: To answer the question "What is the sentiment of this Selected Content between Positive or Negative?", we need to know: "How does the guest evaluate the convenience and surroundings of the hotel's location?"

    Example 3: 
    Q: Content: "The breakfast prepared by Richard was delicious and freshly prepared." What is the sentiment of this Selected Content between Positive or Negative?
    Decomposition: To answer the question "What is the sentiment of this Selected Content between Positive or Negative?", we need to know: "What are the specific details of the guest's feedback regarding the quality and preparation of the breakfast?"

    Example 4: Q: Content: "the building is completely refurbished, it is confortable" What is the sentiment of this Selected Content between Positive or Negative?

    Decomposition:To answer the question "What is the sentiment of this Selected Content between Positive or Negative?", we need to know "How does the guest describe the physical condition and comfort level of the hotel building?"

    Example 5: Q: Content: "We had a late check in and there was no one in the hotel to give us keys and the door was locked."

    Decomposition:To answer the question "What is the sentiment of this Selected Content between Positive or Negative?", we need to know:"What specific obstacles did the guest face during the check-in process and was there any staff assistance available?"

    Current Task:
    Q: Content: "{content}"
    Decomposition:"""

    try:
        res1 = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg_1}],
            temperature=0.1 
        ).choices[0].message.content
        
        sub_q = extract_sub_q(res1)
        
        if not sub_q:
            return "DECOMPOSITION_FAILED", "N/A", "FAILED_AT_STAGE_1"
            
    except Exception as e:
        return f"ERROR_ST1: {str(e)}", "N/A", "FAILED_AT_STAGE_1"

# --- รอบที่ 2: Stage 2.1 - Subproblem Solving ---
    prompt_2_1 = f"""Instruction: Answer the sub-question briefly based on the review content.

    Example 1:
    Content: "There was a problem with heating water at that time and we can understand, but eventually we couldn't manage to have a proper shower with hot water."
    Sub-question: What was the guest's experience with the hotel's basic facilities (hot water) and was it resolved?
    Answer: The guest experienced a persistent problem with the hot water facility which remained unresolved, preventing them from having a proper shower.

    Current Task:
    Content: "{content}"
    Sub-question: {sub_q}
    Answer:"""
    
    res2 = client.chat.completions.create(
        model=MODEL_NAME, messages=[{"role": "user", "content": prompt_2_1}], temperature=0.1
    ).choices[0].message.content
    ans1 = clean_ai_output(res2)

    # --- รอบที่ 3: Stage 2.2 - Final Solving  ---
    # (1) Constant Example (2) Previous Q&A (3) Next Question
    prompt_2_2 = f"""Instruction: Based on the content and previous analysis, answer the final question.

    Example 1:
    Content: "There was a problem with heating water at that time and we can understand, but eventually we couldn't manage to have a proper shower with hot water."
    Previous Q&A:
    Q: What was the guest's experience with the hotel's basic facilities (hot water) and was it resolved?
    A: The guest experienced a persistent problem with the hot water facility which remained unresolved, preventing them from having a proper shower.
    Next Question: What is the sentiment of this Selected Content between Positive or Negative?
    Answer: Negative

    Current Task:
    Content: "{content}"
    Previous Q&A:
    Q: {sub_q}
    A: {ans1}
    Next Question: What is the sentiment of this Selected Content between Positive or Negative?
    Answer:"""

    res3 = client.chat.completions.create(
        model=MODEL_NAME, messages=[{"role": "user", "content": prompt_2_2}], 
        temperature=0.6, top_p=0.95 
    ).choices[0].message.content
    final_raw = clean_ai_output(res3)
    
    return sub_q, ans1, final_raw

# %% [4] เริ่มประมวลผล
df = pd.read_csv(INPUT_CSV_PATH)
results = []

print(f"🚀 เริ่มรัน LtoM Integrated Sequential Pipeline (Plain Text Mode) สำหรับ {len(df)} แถว...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    content = row['Selected Content']
    
    # รัน 3 ขั้นตอนสะสมบริบท
    sub_q, evidence, raw_final = run_lto_m_expert_sequential(content)
    final_sentiment = parse_label(raw_final)
    
    results.append({
        "ID": row['ID'],
        "Content": content,
        "Sub_Question": sub_q,
        "Evidence": evidence,
        "Final_Sentiment": final_sentiment,
        "Raw_Stage2_2": raw_final
    })

    if (idx + 1) % 10 == 0:
        pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
print(f"✅ เสร็จสมบูรณ์! ผลลัพธ์อยู่ที่: {OUTPUT_PATH}")
