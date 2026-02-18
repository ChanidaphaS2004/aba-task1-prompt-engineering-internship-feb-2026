#Code สำหรับPromt: Least to Most Stage2 Solve Sub-question Ver.2
#ปรับ Promt ต่างจาก Ver.1 โดยเพิ่มตัวอย่างการตอบSub-question 
#แก้ไขปัญหา infinite loop
# ---------------------------------------------------
# %% [1] การตั้งค่า Path และโมเดล
import os
import re
import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm
INPUT_STAGE1_PATH = "../Result/Sentiment_All_Results_L2M_Stage1_ver2EP2_1.5b.csv" 
OUTPUT_STAGE2_PATH = "../Result/Sentiment_All_Results_L2M_Stage2_ver2_1.5b.csv"

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama", 
)
MODEL_NAME = "deepseek-r1:1.5b"

# %% [2] ฟังก์ชันช่วยเหลือ (Cleaning & Parsing)
def call_model(prompt, temp):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp
        )
        full_text = response.choices[0].message.content
        return re.sub(r'<think>.*?</think>', '', full_text, flags=re.DOTALL).strip()
    except Exception as e:
        return f"Error: {str(e)}"

def parse_sentiment(text):
    """สกัดผลลัพธ์จากประโยคสุดท้าย"""
    lines = text.strip().split('\n')
    last_line = lines[-1] if lines else text
    if re.search(r'\bPositive\b', last_line, re.I): return "Positive"
    if re.search(r'\bNegative\b', last_line, re.I): return "Negative"
    return "Unknown"

# %% [3] ฟังก์ชันหลักสำหรับการแก้ปัญหาแบบ "มีแค่ไหนใช้แค่นั้น"
def run_l2m_solving_stage(content, q1, q2):
    ans1 = ""
    ans2 = ""
    accumulated_qa = ""

    # --- Step 2.1: Solve Sub-question 1 (ถ้ามีข้อมูล) ---
    if pd.notna(q1) and str(q1).strip():
        prompt_1 = f"""Instruction: Answer the sub-question briefly based on the review content.
        Example:
        Q: Content: "There was a problem with heating water at that time and we can understand, but eventually we couldn't manage to have a proper shower with hot water."
        Sub_Question1: Did the guest encounter a problem with the hot water system?
        Answer_SubQ1: The content states "There was a problem with heating water at that time". This indicates that the guest's experience was problematic due to a functional failure of the water heating system.

        Content: "{content}"
        Sub_Question1: {q1}
        Answer_SubQ1:"""
        ans1 = call_model(prompt_1, temp=0.1)
        accumulated_qa += f"Sub_Question1: {q1}\nAnswer_SubQ1: {ans1}\n"

    # --- Step 2.2: Solve Sub-question 2 (ถ้ามีข้อมูล + ใช้ความรู้สะสม) ---
    if pd.notna(q2) and str(q2).strip():
        prompt_2 = f"""Instruction: Based on the content and previous analysis, answer the next sub-question.
        Example:
        Q: Content: "There was a problem with heating water at that time and we can understand, but eventually we couldn't manage to have a proper shower with hot water."
        Previous Q&A:
        Sub_Question1: What was the guest's experience with the hotel's basic facilities (hot water)?
        Answer_SubQ1: The content states "There was a problem with heating water at that time". This indicates that the guest's experience was problematic due to a functional failure of the water heating system.

        Next Sub_Question:
        Sub_Question2: Was the hot water issue resolved before the guest finished their stay?
        Answer_SubQ2: No, the text confirms that eventually they still could not have a proper shower, meaning it was not resolved.

        Content: "{content}"
        """
        if accumulated_qa:
            prompt_2 += f"\nPrevious Q&A:\n{accumulated_qa}"
        
        prompt_2 += f"\nNext Question:\nSub_Question2: {q2}\nAnswer_SubQ2:"
        ans2 = call_model(prompt_2, temp=0.1)
        accumulated_qa += f"Sub_Question2: {q2}\nAnswer_SubQ2: {ans2}\n"

    # --- Step 2.3: Final Synthesis (สรุปจากข้อมูลที่มีทั้งหมด) ---
    main_q = "What is the sentiment of this Selected Content between Positive or Negative?"
    prompt_final = f"""Instruction: Based on the content and all available analysis, answer the final question.
    Example:
        Q: Content: "There was a problem with heating water at that time and we can understand, but eventually we couldn't manage to have a proper shower with hot water."
        Previous Q&A:
        Sub_Question1: Did the guest encounter a problem with the hot water system?
        Answer_SubQ1: The content states "There was a problem with heating water at that time". This indicates that the guest's experience was problematic due to a functional failure of the water heating system.
        Sub_Question2: Was the hot water issue resolved before the guest finished their stay?
        Answer_SubQ2: No, the text confirms that eventually they still could not have a proper shower, meaning it was not resolved.

        Q: What is the sentiment of this Selected Content between Positive or Negative? 
        Answer: We know that the guest's experience with the hot water was problematic. We also know that the issue was not resolved before the guest finished their stay. Therefore, the failure to meet basic expectations results in a negative experience. The answer is Negative.

    Content: "{content}"
    """
    if accumulated_qa:
        prompt_final += f"\nPrevious Q&A:\n{accumulated_qa}"
    
    prompt_final += f"""
    Next Question: {main_q}
    Note: Start your answer with "A: We know that..." and end with "The answer is [Positive/Negative]."
    Answer:"""
    
    final_synthesis = call_model(prompt_final, temp=0.6)
    return ans1, ans2, final_synthesis

# %% [4] เริ่มกระบวนการรัน
df = pd.read_csv(INPUT_STAGE1_PATH)
results = []

print(f"🚀 เริ่มรัน Stage 2 (Sequential Solving) - Model: {MODEL_NAME}")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    content = row['Selected Content']
    q1 = row['Sub_Question_1']
    q2 = row['Sub_Question_2']
    
    # รัน Sequential 3-Step Process
    ans1, ans2, synthesis = run_l2m_solving_stage(content, q1, q2)
    
    # สกัดขั้วอารมณ์จากประโยคสรุป
    sentiment = parse_sentiment(synthesis)
    
    results.append({
        "ID": row['ID'],
        "Content": content,
        "Sub_Q1": q1,
        "Ans1": ans1,
        "Sub_Q2": q2,
        "Ans2": ans2,
        "Final_Synthesis": synthesis,
        "Predicted_Sentiment": sentiment
    })
    
    if (idx + 1) % 5 == 0:
        pd.DataFrame(results).to_csv(OUTPUT_STAGE2_PATH, index=False, encoding='utf-8-sig')

pd.DataFrame(results).to_csv(OUTPUT_STAGE2_PATH, index=False, encoding='utf-8-sig')
print(f"\n✅ Stage 2 เสร็จสมบูรณ์! ผลลัพธ์อยู่ที่: {OUTPUT_STAGE2_PATH}")