# %% [1] การตั้งค่าเบื้องต้น
import os
import re
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

INPUT_CSV_PATH = "../Data/data_sentiment_no_Off.csv" 
OUTPUT_PATH = "../Result/Sentiment_All_Results_LtoM_v2_1.5b.csv"

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL_NAME = "deepseek-r1:1.5b"

def call_model(prompt, temp):
    """ฟังก์ชันกลางเรียก LLM และลบส่วน <think>"""
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
    """สกัดผลลัพธ์ Positive/Negative จาก Synthesis"""
    lines = text.strip().split('\n')
    last_line = lines[-1] if lines else text
    if re.search(r'\bPositive\b', last_line, re.I): return "Positive"
    if re.search(r'\bNegative\b', last_line, re.I): return "Negative"
    return "Unknown"


# %% [2] ฟังก์ชัน Sequential Pipeline (Stage 1 -> 2.1 -> 2.2)
def parse_decomposition_output(full_output):
    clean_text = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL).strip()
    def clean_string(text):
        text = re.sub(r'\[/?SUB_Q\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^[,\d\s\.\-\*]+', '', text).strip()
        return text.replace('"', '').replace("'", "").strip()

    found_questions = []
    tags_found = re.findall(r'\[SUB_Q\](.*?)\[/SUB_Q\]', clean_text, re.IGNORECASE | re.DOTALL)
    if tags_found:
        found_questions = tags_found
    if len(found_questions) < 2:
        q_pattern = r'([^?!\n]*(?:What|How|Who|Where|When|Why|Is|Was|Are|Were|Does|Did|Can|Could)[^?!\n]+(?:\?|$))'
        all_potential = re.findall(q_pattern, clean_text, re.IGNORECASE)
        for q in all_potential:
            q_c = clean_string(q)
            if "sentiment" not in q_c.lower() and len(q_c) > 15:
                if q_c not in [clean_string(fq) for fq in found_questions]:
                    found_questions.append(q)
    results = [clean_string(q) for q in found_questions]
    while len(results) < 2: results.append("")
    return results[:2]

def run_decomposition(content):
    # --- รอบที่ 1: Stage 1 - Decomposition  ---
    system_msg = "You are a specialized Sentiment Analysis expert."
    
    user_msg = f"""Instruction: Decompose the hotel review into 2 sequential sub-questions. Use Hierarchical Logic: Fact first, then Outcome. No 'and' allowed.
    "RULES:"
        "1. DO NOT use the word 'and' in any sub-question.\n"
        "2. Sub_Q1 (Fact): Identify the initial incident or situation.\n"
        "3. Sub_Q2 (Outcome): Identify the resolution or the final impact.\n"
        "4. DO NOT determine if it is Positive or Negative."
        "5. ONLY output 2 sub-questions."
        "6. Each sub-question must be wrapped in [SUB_Q] tags."

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
    Q: Content: "{content}"
    Decomposition:"""
    full_prompt = system_msg + "\n\n" + user_msg
    return call_model(full_prompt, temp=0.6)

# %% [3] ฟังก์ชัน Stage 2: Sequential Solving Logic
def run_l2m_solving_stage(content, q1, q2):
    ans1, ans2, accumulated_qa = "", "", ""

    # Step 2.1: Fact Check
    if q1:
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

    # Step 2.2: Outcome (Accumulated Knowledge)
    if q2:
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

    # Step 2.3: Synthesis
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

# %% [4] เริ่มประมวลผล
if not os.path.exists('result'): os.makedirs('result')

df_input = pd.read_csv(INPUT_CSV_PATH) #.head(30) 
final_results = []

print(f"🚀 เริ่มต้นการประมวลผล L2M Full Process (Model: {MODEL_NAME})")

for idx, row in tqdm(df_input.iterrows(), total=len(df_input)):
    content = row['Selected Content']
    
    # --- Stage 1: Decomposition ---
    raw_stage1 = run_decomposition(content)
    q1, q2 = parse_decomposition_output(raw_stage1)
    
    # --- Stage 2: Solving ---
    a1, a2, synthesis = run_l2m_solving_stage(content, q1, q2)
    sentiment = parse_sentiment(synthesis)
    
    final_results.append({
        "ID": row['ID'],
        "Content": content,
        "Sub_Q1": q1,
        "Ans1": a1,
        "Sub_Q2": q2,
        "Ans2": a2,
        "Final_Synthesis": synthesis,
        "Predicted_Sentiment": sentiment
    })
    
    if (idx + 1) % 5 == 0:
        pd.DataFrame(final_results).to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

pd.DataFrame(final_results).to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
print(f"✅ ประมวลผลเสร็จสิ้น! บันทึกที่: {OUTPUT_PATH}")
