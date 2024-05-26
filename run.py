import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as genai

def get_api_key():
    GOOGLE_API_KEY='your api key'
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

tokenizer = AutoTokenizer.from_pretrained("NTQAI/chatntq-ja-7b-v1.0")
model = AutoModelForCausalLM.from_pretrained(
  "NTQAI/chatntq-ja-7b-v1.0",
  torch_dtype="auto",
)
model.eval()

if torch.cuda.is_available():
    model = model.to("cuda")

def build_prompt(user_query):
    sys_msg = "あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。"
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

out = local_LLM("こんにちは")
print(out)
