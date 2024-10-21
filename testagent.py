import os
import json
import random
from langchain.agents import initialize_agent,Tool,AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.agent_toolkits.load_tools import load_tools 
from langchain_openai import OpenAI
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
import ast

# LLMの初期化
llm = OpenAI(temperature=0.7)

def process_text_to_numbers(input_text):
    """
    入力された文字列をひらがなと数字の辞書に基づいて数値化する関数。
    辞書に存在しないひらがなは0に置き換えられる。

    Args:
        input_text (str): 数値化したい文字列

    Returns:
        list: 入力文字列中のひらがなを数値に変換したリスト。
              辞書に存在しないひらがなは0に置き換えられる。
    """
    hiragana = [
        'あ', 'い', 'う', 'え', 'お',
        'か', 'き', 'く', 'け', 'こ',
        'さ', 'し', 'す', 'せ', 'そ',
        'た', 'ち', 'つ', 'て', 'と',
        'な', 'に', 'ぬ', 'ね', 'の',
        'は', 'ひ', 'ふ', 'へ', 'ほ',
        'ま', 'み', 'む', 'め', 'も',
        'や', 'ゆ', 'よ',
        'ら', 'り', 'る', 'れ', 'ろ',
        'わ', 'を', 'ん',
        'が', 'ぎ', 'ぐ', 'げ', 'ご',
        'ざ', 'じ', 'ず', 'ぜ', 'ぞ',
        'だ', 'ぢ', 'づ', 'で', 'ど',
        'ば', 'び', 'ぶ', 'べ', 'ぼ',
        'ぱ', 'ぴ', 'ぷ', 'ぺ', 'ぽ'
    ]

    numbers = list(range(1, 77))
    random.shuffle(numbers)

    hiragana_dict = {char: numbers[i] for i, char in enumerate(hiragana)}

    # 入力テキストの処理
    converted_list = []
    for char in input_text:
        if char in hiragana_dict:
            converted_list.append(hiragana_dict[char])
        else:
            # 辞書に存在しないひらがなは0にする。
            # ひらがな以外の文字については、この例では0として扱います。
            converted_list.append(0)

    return converted_list

def rsa_encrypt_decrypt(messages):
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def mod_inverse(e, phi):
        d = 0
        x1 = 0
        x2 = 1
        y1 = 1
        temp_phi = phi

        while e > 0:
            temp1 = temp_phi // e
            temp2 = temp_phi - temp1 * e
            temp_phi = e
            e = temp2

            x = x2 - temp1 * x1
            y = d - temp1 * y1

            x2 = x1
            x1 = x
            d = y1
            y1 = y

        if temp_phi == 1:
            return d + phi

    def modular_exponentiation(base, exp, n):
        result = 1
        base = base % n
        while exp > 0:
            if exp % 2 == 1:
                result = (result * base) % n
            base = (base * base) % n
            exp //= 2
        return result
    p = 7
    q = 11
    n = p*q
    phi = (p - 1) * (q - 1)

    e = next(i for i in range(2, phi) if gcd(i, phi) == 1)
    d = mod_inverse(e, phi)

    public_key = (n, e)
    private_key = (n, d)
    # 各メッセージを暗号化
   # 数値化された結果が整数型であることを確認
    try:
        actual_list = ast.literal_eval(messages)
        # 変換されたリストを出力
        print(f"Actual List: {actual_list}")

        # リストの各要素を処理 (以下の行は例です)
        # 'e' と 'n' は事前に定義されているべきです
        ciphertexts = [modular_exponentiation(message, e, n) for message in actual_list]
        print(f"Ciphertexts: {ciphertexts}")

    except (SyntaxError, ValueError) as e:
        print("リストへの変換に失敗しました。", e)
    
    return ciphertexts


tools = [
    Tool(name="数値変換", description="入力された文字列を、与えられたひらがな-数字辞書に基づいて数値化する。", func=process_text_to_numbers),
    Tool(name="暗号化", description=" ある数をある規則に従って暗号化する", func=rsa_encrypt_decrypt), 
]


def read_file(file_path: str) -> str:
    """ファイルを読み込み、内容を文字列として返す"""
    with open(file_path, "r") as f:
        return f.read()

class ReadFileInput(BaseModel):
    """Input for ReadFile tool."""
    file_path: str = Field(description="読み込むファイルのパス")

tools.append(StructuredTool(
    name="ファイル読み込み",
    description="指定されたパスにあるファイルの内容を読み込みます。",
    func=read_file,
    args_schema=ReadFileInput
)) 

# エージェントの初期化
memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True, 
    memory=memory
)

# ユーザーからの文字列入力
input_string = input("文字列を入力してください: ")


# 入力文字列をJSON形式で保存
input_data = {"text": input_string}
with open("input.json", "w", encoding="utf-8") as f:
    json.dump(input_data, f, ensure_ascii=False)

# エージェントにタスクを指示
result = agent.run("""入力された文字を暗号化してください。次の手順に従ってください。
1. "input.json" というファイルを読み込み、文字列を取得します。
2.取得した文字列を数値変換ツールで数値のリストとして取得します。
3.次にリストを暗号化ツールを用いて暗号化します。""")
print(f"実行結果: {result}")