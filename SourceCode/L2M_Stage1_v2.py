#Code สำหรับPromt: Least to Most Stage1 Decomposition Ver.2
#ปรับ Promt ต่างจาก Ver.1 เพิ่ม Tag [SUB_Q]และไม่ให้มี and ในSub-question
#ปรับฟังก์ชันสกัด Sub-questions 
# ---------------------------------------------------
# %% [1] @title 1. Library และการตั้งค่า Local Path
import os
import re
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

INPUT_CSV_PATH = "./data/data_sentiment_no_Off.csv" 
OUTPUT_STAGE1_PATH = "./result/Sentiment_All_Results_L2M_Stage1_ver2EP2_1.5b.csv"

# %% [2] @title 2. เชื่อมต่อ Ollama (Local)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama", 
)

MODEL_NAME = "deepseek-r1:1.5b" 
print(f"✅ เชื่อมต่อ Ollama (Model: {MODEL_NAME}) สำหรับ Stage 1")

# %% [3] @title 3. ฟังก์ชันสกัด Sub-questions 
import re

def parse_decomposition_output(full_output):
    """
    สกัด 2 sub-questions โดยเน้นดักจับ [SUB_Q] tags ก่อน 
    ถ้าไม่มีค่อยใช้ Wh-words waterfall
    """
    clean_text = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL).strip()
    
    def clean_string(text):
        text = re.sub(r'\[/?SUB_Q\]', '', text, flags=re.IGNORECASE) 
        text = re.sub(r'^[,\d\s\.\-\*]+', '', text).strip()
        return text.replace('"', '').replace("'", "").strip()

    # --- Waterfall Logic ---
    found_questions = []
    
    # แบบที่ 1: ดึงจาก [SUB_Q]...[/SUB_Q] 
    tags_found = re.findall(r'\[SUB_Q\](.*?)\[/SUB_Q\]', clean_text, re.IGNORECASE | re.DOTALL)
    if tags_found:
        found_questions = tags_found

    # แบบที่ 2: ถ้าไม่มี Tag ให้ใช้ Wh-words waterfall 
    if len(found_questions) < 2:
        q_pattern = r'([^?!\n]*(?:What|How|Who|Where|When|Why|Is|Was|Are|Were|Does|Did|Can|Could)[^?!\n]+(?:\?|$))'
        all_potential = re.findall(q_pattern, clean_text, re.IGNORECASE)
        for q in all_potential:
            q_c = clean_string(q)
            if "sentiment" not in q_c.lower() and len(q_c) > 15:
                if q_c not in [clean_string(fq) for fq in found_questions]:
                    found_questions.append(q)

    results = [clean_string(q) for q in found_questions]
    while len(results) < 2:
        results.append("")
    return results[:2]

def run_decomposition(content):
    system_msg = (
        "You are a specialized Sentiment Analysis expert. Your task is to analyze a hotel review "
        "and identify the specific information (sub-question) needed to determine the final sentiment."
        "RULES:"
        "1. DO NOT use the word 'and' in any sub-question.\n"
        "2. Sub_Q1 (Fact): Identify the initial incident or situation.\n"
        "3. Sub_Q2 (Outcome): Identify the resolution or the final impact.\n"
        "4. DO NOT determine if it is Positive or Negative."
        "5. ONLY output 2 sub-questions."
        "6. Each sub-question must be wrapped in [SUB_Q] tags."
    )
    
    user_msg = f"""Instruction: Decompose the hotel review into 2 sequential sub-questions. Use Hierarchical Logic: Fact first, then Outcome. No 'and' allowed.

    Example 1:
    Q: Content: "There was a problem with heating water at that time and we can understand, but eventually we couldn't manage to have a proper shower with hot water."What is the sentiment of this Selected Content between Positive or Negative?
    Decomposition: To answer the question “What is the sentiment of this Selected Content?”, we need to know: 
    [SUB_Q]Did the guest encounter a problem with the hot water system?[/SUB_Q] 
    [SUB_Q]Was the hot water issue resolved before the guest finished their stay?[/SUB_Q]

    Example 2:
    Q: Content: "Nice location, not far from Airport and Mackenzie beach."What is the sentiment of this Selected Content between Positive or Negative?
    Decomposition: To answer the question “What is the sentiment of this Selected Content?”, we need to know: 
    [SUB_Q]Is the hotel located near the airport?[/SUB_Q] 
    [SUB_Q]Does the location provide a convenient experience for the guest?[/SUB_Q]

    Example 3:
    Q: Content: "The breakfast prepared by Richard was delicious and freshly prepared."What is the sentiment of this Selected Content between Positive or Negative?
    Decomposition: To answer the question “What is the sentiment of this Selected Content?”, we need to know: 
    [SUB_Q]Was the breakfast freshly prepared?[/SUB_Q] 
    [SUB_Q]Did the guest find the taste delicious?[/SUB_Q]

    Example 4:
    Q: Content: "the building is completely refurbished, it is confortable"What is the sentiment of this Selected Content between Positive or Negative?
    Decomposition: To answer the question “What is the sentiment of this Selected Content?”, we need to know: 
    [SUB_Q]Is the building completely refurbished?[/SUB_Q] 
    [SUB_Q]Does the guest find the stay comfortable?[/SUB_Q]

    Example 5:
    Q: Content: "We had a late check in and there was no one in the hotel to give us keys and the door was locked."What is the sentiment of this Selected Content between Positive or Negative?
    Decomposition: To answer the question “What is the sentiment of this Selected Content?”, we need to know: 
    [SUB_Q]Did the guest face a locked door during late check-in?[/SUB_Q] 
    [SUB_Q]Was there any staff assistance provided to the guest?[/SUB_Q]

    Current Task:
    Q: Content: "{content}" What is the sentiment of this Selected Content between Positive or Negative?
    Decomposition:"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.6,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e: 
        return f"Error: {str(e)}"

# %% [4] @title 4. เริ่มประมวลผล Stage 1 
df_input = pd.read_csv(INPUT_CSV_PATH) #.head(30)
stage1_results = []

print(f"🚀 เริ่มรัน Stage 1 (Decomposition) - Model: {MODEL_NAME}")

for idx, row in tqdm(df_input.iterrows(), total=len(df_input)):
    content = row['Selected Content']
    
    raw_output = run_decomposition(content)
    # รับค่ากลับมาเป็น List 2 สมาชิก
    sub_qs = parse_decomposition_output(raw_output)

    stage1_results.append({
        "ID": row['ID'],
        "Selected Content": content,
        "Sub_Question_1": sub_qs[0], 
        "Sub_Question_2": sub_qs[1], 
        "Raw_Stage1_Output": raw_output
    })

    if (idx + 1) % 5 == 0:
        pd.DataFrame(stage1_results).to_csv(OUTPUT_STAGE1_PATH, index=False, encoding='utf-8-sig')

df_final = pd.DataFrame(stage1_results)
df_final.to_csv(OUTPUT_STAGE1_PATH, index=False, encoding='utf-8-sig')

print(f"\n✅ Stage 1 เสร็จสมบูรณ์! แยก Sub-Questions เป็น 2 คอลัมน์เรียบร้อย")
print(f"📁 บันทึกที่: {OUTPUT_STAGE1_PATH}")