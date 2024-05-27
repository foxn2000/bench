import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as genai

def get_api_key():
    GOOGLE_API_KEY='your API key'
    return GOOGLE_API_KEY

GOOGLE_API_KEY = get_api_key()
genai.configure(api_key=GOOGLE_API_KEY)
model_gemini = genai.GenerativeModel('gemini-pro')

def con(call_in):
    response = model_gemini.generate_content(str(call_in))
    list_deta = response.candidates[0].content.parts
    res = ""
    for i in list_deta:
        res = i.text
    return res

tokenizer = AutoTokenizer.from_pretrained("DataPilot/ArrowPro-7B-KUJIRA")
model = AutoModelForCausalLM.from_pretrained(
  "DataPilot/ArrowPro-7B-KUJIRA",
  torch_dtype="auto",
)
model.eval()

if torch.cuda.is_available():
    model = model.to("cuda")

def build_prompt(user_query):
    sys_msg = "あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。また考える過程も出力してください。"
    template = """[INST] <<SYS>>
{}
<</SYS>>

{}[/INST]"""
    return template.format(sys_msg,user_query)

def local_LLM(user_inputs_text):
    user_inputs = {
        "user_query": user_inputs_text,
    }
    prompt = build_prompt(**user_inputs)
    input_ids = tokenizer.encode(
        prompt, 
        add_special_tokens=True, 
        return_tensors="pt"
    )
    tokens = model.generate(
        input_ids.to(device=model.device),
        max_new_tokens=500,
        temperature=1,
        top_p=0.95,
        do_sample=True,
    )
    out = tokenizer.decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    return out

def make_input(LLM_output,question,Correct_text,eval_aspect,):
    exam_text = f"""問題, 正解例, 採点基準, 言語モデルが生成した回答が与えられます。

# 指示
「採点基準」と「正解例」を参考にして、回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。

# 問題
{question}

# 正解例
{Correct_text}

# 採点基準
基本的な採点基準
- 1点: 誤っている、 指示に従えていない
- 2点: 誤っているが、方向性は合っている
- 3点: 部分的に誤っている、 部分的に合っている
- 4点: 合っている
- 5点: 役に立つ

基本的な減点項目
- 不自然な日本語: -1点
- 部分的に事実と異なる内容を述べている: -1点
- 「倫理的に答えられません」のように過度に安全性を気にしてしまっている: 2点にする

問題固有の採点基準
{eval_aspect}

# 言語モデルの回答
{LLM_output}

# ここまでが'言語モデルの回答'です。回答が空白だった場合、1点にしてください。

# 指示
「採点基準」と「正解例」を参考にして、回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。
"""
    return exam_text
count = 0

with open('test.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if count > 0:
            print(f'問題: {row[0]}, 回答: {row[1]}, 採点ポイント: {row[2]}')
            out = local_LLM(str(row[0]))
            exam_text = make_input(out,row[0],row[1],row[2])
            res = con(exam_text)
            with open("result.txt", mode='a', encoding="utf-8") as f:
                f.write(str(res) +"\n")
        else:
            count = count + 1
            print(count)


