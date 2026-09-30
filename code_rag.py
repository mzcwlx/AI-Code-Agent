import os
import ast
import ollama
import numpy as np
import faiss


def collect_python_files(root="."):
    python_files = []

    for current_root, dirs, files in os.walk(root):
        dirs[:] = [
            d for d in dirs
            if d not in {".git", "__pycache__", ".venv"}
        ]

        for file in files:
            if file.endswith(".py"):
                path = os.path.join(current_root, file)
                python_files.append(path)

    return python_files


def read_code_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_code(text, file):
    tree = ast.parse(text)
    lines = text.splitlines()

    chunks = []

    for node in tree.body:

        # 函数
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):

            start = node.lineno - 1
            end = node.end_lineno

            code = "\n".join(lines[start:end])

            chunks.append({
                "file": file,
                "type": "function",
                "name": node.name,
                "code": code
            })

        # 非函数的顶层代码
        else:

            start = node.lineno - 1
            end = node.end_lineno

            code = "\n".join(lines[start:end])

            chunks.append({
                "file": file,
                "type": "module",
                "name": "module",
                "code": code
            })

    return chunks

def embed_code(chunks):
    embeddings = []
    for chunk in chunks:
        embedding = ollama.embed(
            model="bge-m3",
            input=chunk["code"]
        )
        embeddings.append(embedding)
    return embeddings

def build_index(all_chunks):
    embeddings = embed_code(all_chunks)
    vectors = np.array(
        [item["embeddings"][0] for item in embeddings],
        dtype="float32"
    )
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(1024)
    index.add(vectors)
    return index

def retrieve_code(query, index, all_chunks, top_k=5):

    query_embedding = ollama.embed(
        model="bge-m3",
        input=query
    )

    query_vector = np.array(
        [query_embedding["embeddings"][0]],
        dtype="float32"
    )

    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        results.append({
            "score": float(score),
            "file": all_chunks[idx]["file"],
            "type": all_chunks[idx]["type"],
            "name": all_chunks[idx]["name"],
            "code": all_chunks[idx]["code"]
        })

    return results
def format_results(results):

    context = ""

    for result in results:
        context += f"""
文件：{result["file"]}
类型：{result["type"]}
名称：{result["name"]}
相似度：{result["score"]}

代码：
{result["code"]}

"""
    
    return context

def build_code_rag(root="."):

    files = collect_python_files(root)

    all_chunks = []

    for file in files:
        content = read_code_file(file)
        chunks = split_code(content, file)
        all_chunks.extend(chunks)

    index = build_index(all_chunks)

    return index, all_chunks


if __name__ == "__main__":
    index, all_chunks = build_code_rag(".")
    query = "add函数有什么问题？"
    results = retrieve_code(query, index, all_chunks, top_k=5)
    context = format_results(results)
    user_message = f"""
用户的问题：

{query}

下面是从项目代码中检索到的相关代码：

{context}

请根据这些代码分析用户的问题。
"""

    print("=" * 50)
    print("最终给 LLM 的内容：")
    print(user_message)

    response = ollama.chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": user_message
        }
    ]
)

    print("=" * 50)
    print("LLM回答：")
    print(response["message"]["content"])
    