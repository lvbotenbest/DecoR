

"""批量请求千帆中控模型脚本，输入文件格式为excel，history与input为必须col"""

import json
import os
import requests
import traceback

prompt_template = "You are a Capability Decomposition Engine. Your task is to decompose the user query into its capability-space representation C(q) = {S, K, D, F}. Follow all rules strictly and output JSON only.\n\n---\n\n1. Skill Set (S):\nIdentify the skills required to answer the query.\n- You may freely generate categories.\n- Examples: reasoning, logical inference, mathematics, coding, translation, writing, information extraction, multi-step planning, role-playing, style imitation, summarization.\n- Output as a list.\nAlso output \"S_reason\": one sentence explaining why these skills are required.\n\n2. Knowledge Domain (K):\nIdentify the knowledge domains needed for the query.\n- Freely generated categories; no fixed list.\n- Examples: general knowledge, medicine, law, finance, computer science, physics, ACG, history, philosophy.\n- If no specific knowledge is required, output \"none\".\n- Output as a list.\nAlso output \"K_reason\": one sentence explaining why these domains are needed.\n\n3. Difficulty / Instruction Complexity (D):\nChoose exactly one:\n- D0 (Trivial): almost no reasoning; direct/simple request.\n- D1 (Simple): mild understanding; single goal; light reasoning.\n- D2 (Moderate): multiple requirements or multi-step tasks; needs organization/judgment.\n- D3 (Hard): complex tasks requiring deep reasoning, planning, abstraction, or structured logic.\nAlso output \"D_reason\": one sentence explaining why this difficulty level matches the query.\n---\n\nIMPORTANT RULES:\n- Output MUST be valid pure JSON.\n- Do NOT include markdown code fences such as ```json or ``` anywhere.\n\n\nOutput Format (STRICT):\n\nReturn ONLY the following JSON structure:\n\n{\n  \"S\": [...],\n  \"S_reason\": \"...\",\n  \"K\": [...],\n  \"K_reason\": \"...\",\n  \"D\": \"D0 or D1 or D2 or D3\",\n  \"D_reason\": \"...\",\n}\n\nNo explanations. No extra text. Only valid JSON.\n"

class QianfanRequest(object):
    """
    调用千帆模型，可参考:https://cloud.baidu.com/doc/qianfan-api/s/3m7of64lb
    """

    def __init__(self):
        """
            初始化
        """
        self.headers = {
            'Content-Type': 'application/json',
            # Set your Qianfan API token via the QIANFAN_API_KEY environment variable.
            'Authorization': 'Bearer ' + os.getenv('QIANFAN_API_KEY', 'YOUR_QIANFAN_TOKEN')
        }
        self.url = "https://qianfan.baidubce.com/v2/chat/completions"
        # self.url = "https://qianfan.baidubce.com/v2/router/chat/completions"

    def req_model(self, model_name, content, is_system=False, is_thinking=False):
        """
        支持千帆平台模型，demo：
            ## ernie-x1-32k、ernie-4.5-8k-preview、ernie-4.0-8k
            ## deepseek-r1、deepseek-v3、deepseek-r1-250528
            ## qwq-32b、qwen3-30b-a3b（"enable_thinking": False）、qwen3-235b-a22b（"enable_thinking": False）
        """

        messages = [{"role": "user", "content": content}]
        if is_system:
            messages = content

        payload = json.dumps({
            "model": model_name,
            "stream": False,
            "messages": messages,
            "temperature": 0.01,
            "top_p": 0.8,
            "max_output_tokens": 1 * 1024,
            "penalty_score": 1.0
        }, ensure_ascii=False)

        if model_name in ("qwen3-30b-a3b", "qwen3-235b-a22b"):
            payload = json.dumps({
                "model": model_name,
                "stream": False,
                "messages": messages,
                "temperature": 0.8,
                "top_p": 0.8,
                "max_output_tokens": 1 * 1024,
                "penalty_score": 1.0,
                "enable_thinking": is_thinking
            }, ensure_ascii=False)

        response = requests.request("POST", self.url, headers=self.headers, data=payload.encode("utf-8"))
       
        return response.text

    def req_qianfan(self, index, prompt, model_name):
        """
            单线程请求大模型
        """

        result = {
            "index": index,
            "answer": "ERR",
            "content": "ERR"
        }
        try:
            answer = self.req_model(model_name=model_name, content=prompt)
            choices = json.loads(answer).get("choices", [{}])
            content = choices[0].get("message", {}).get("content", "ERR") if choices else "ERR"
            result.update({
                # "answer": answer,
                "content": content
            })

        except Exception as e:  # 捕获异常并绑定到变量e
            traceback.print_exc()
            result["error"] = str(e)
        return result

    def req_decomposition(self, index, prompt, model_name,query):
        """
            单线程请求大模型
        """

        result = {
            "index": index,
            "prompt": query,
            "content": "ERR",
            "input_tokens": 0,
            "output_tokens": 0,
        }
        try:
            answer = self.req_model(model_name=model_name, content=prompt)
            choices = json.loads(answer).get("choices", [{}])
            content = choices[0].get("message", {}).get("content", "ERR") if choices else "ERR"
            # print(answer)
            input_tokens = json.loads(answer).get("usage", {}).get("prompt_tokens", 0)
            output_tokens = json.loads(answer).get("usage", {}).get("completion_tokens", 0) 
            result.update({
                # "answer": answer,
                "content": content, 
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            })

        except Exception as e:  # 捕获异常并绑定到变量e
            traceback.print_exc()
            result["error"] = str(e)
        return result

    def req_qianfan_batch(self, content, model_name, task_name, max_workers=1):
        """
            批量请求大模型
        """


        try:
            result = self.req_qianfan(0,content, model_name)
        except:
            print(f"KeyError: 索引 失败")


        return result

    def req_qianfan_sample(self, prompt, model_name):
        """
            单线程请求大模型（简化版，直接返回content）
        """
        try:
            answer = self.req_model(model_name=model_name, content=prompt)
            choices = json.loads(answer).get("choices", [{}])
            content = choices[0].get("message", {}).get("content", "ERR") if choices else "ERR"
            return content
        except Exception as e:  # 捕获异常并绑定到变量e
            traceback.print_exc()
            return "ERR"

if __name__ == '__main__':


    # for index in range(len(queries)):
    #     query = queries[index]
    #     history = histories[index]

    #     content = 

    #     result = QianfanRequest().req_qianfan_batch(content, "ipgsn55l_datav60_8b_all_8k", 'eval', max_workers=5)

    #     print(result)

    query = "from typing import List def has_close_elements(numbers: List[float], threshold: float) -> bool: Check if in given list of numbers, are any two numbers closer to each other than given threshold. >>> has_close_elements([1.0, 2.0, 3.0], 0.5) False >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True"
    content = prompt_template+"\n"+ query

    # result = QianfanRequest().req_qianfan(index=0,prompt=content, model_name="grt8bohz_test")
    result = QianfanRequest().req_decomposition(index=0,prompt=content, model_name="grt8bohz_test",query=query)

    print(result)
