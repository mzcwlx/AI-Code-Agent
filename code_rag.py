import os
import ast
import re
import numpy as np
import faiss


from llm_client import client


EMBEDDING_MODEL = "embedding-3"


MAX_CHARS = 2400


EMBED_BATCH_SIZE = 8

MAX_CANDIDATES = 240

GENERATED_OR_DATA_NAMES = {
    "_emoji_codes.py",
    "emoji_codes.py",
}

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "node_modules",
    "dist",
    "build",
    "site-packages",
}


def collect_python_files(root="."):
    """收集适合代码检索的 Python 文件。"""
    python_files = []

    for current_root, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

        for file in files:
            if not file.endswith(".py"):
                continue

            # 过滤明显的生成数据文件。
            if file in GENERATED_OR_DATA_NAMES:
                continue

            path = os.path.join(current_root, file)

            try:
                size = os.path.getsize(path)
            except OSError:
                continue

            # 超大的 Python 文件优先避免进入第一轮 RAG。
            # 正常源文件一般远小于这个值。
            if size > 500_000:
                continue

            python_files.append(path)

    return python_files


def read_code_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _node_code(lines, node):
    start = node.lineno - 1
    end = node.end_lineno
    return "\n".join(lines[start:end])


def _split_long_code(code, max_chars=MAX_CHARS):
    """只为 embedding 切块，不修改真实源代码。"""
    if len(code) <= max_chars:
        return [code]

    lines = code.splitlines()
    chunks = []
    current = []
    current_len = 0

    for line in lines:
        # 一行本身过长时硬切。
        if len(line) > max_chars:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0

            for i in range(0, len(line), max_chars):
                chunks.append(line[i:i + max_chars])
            continue

        line_len = len(line) + 1

        if current and current_len + line_len > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0

        current.append(line)
        current_len += line_len

    if current:
        chunks.append("\n".join(current))

    return chunks


def split_code(text, file):
    """
    优先按 AST 的 class / function / method 切分。
    顶层巨型 module 不再整体进入 embedding，而是继续按行切。
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # 某些仓库的代码可能暂时无法被 AST 解析。
        return [
            {
                "file": file,
                "type": "text",
                "name": "text",
                "code": part,
            }
            for part in _split_long_code(text)
            if part.strip()
        ]

    lines = text.splitlines()
    chunks = []

    def add_node(node, node_type, name):
        code = _node_code(lines, node)
        for index, part in enumerate(_split_long_code(code)):
            chunks.append(
                {
                    "file": file,
                    "type": node_type,
                    "name": name if index == 0 else f"{name} [part {index + 1}]",
                    "code": part,
                }
            )

    # 顶层结构。
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            add_node(node, "class", node.name)

            # 同时单独索引 class 内的方法。
            for child in node.body:
                if isinstance(
                    child,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    add_node(child, "method", f"{node.name}.{child.name}")

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_node(node, "function", node.name)

        else:
            code = _node_code(lines, node)
            for index, part in enumerate(_split_long_code(code)):
                if part.strip():
                    chunks.append(
                        {
                            "file": file,
                            "type": "module",
                            "name": "module"
                            if index == 0
                            else f"module [part {index + 1}]",
                            "code": part,
                        }
                    )

    return chunks


def _tokens(text):
    """
    用于 embedding 前的轻量候选筛选。
    不负责最终语义检索，只负责把几千 chunk 缩小到几百个。
    """
    return set(
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)
    )


def _lexical_score(query, chunk):
    query_tokens = _tokens(query)

    if not query_tokens:
        return 0.0

    searchable = (
        f'{chunk["file"]} '
        f'{chunk["name"]} '
        f'{chunk["code"]}'
    ).lower()

    score = 0.0

    for token in query_tokens:
        if token in searchable:
            score += 1.0

            # 文件名 / 函数名命中比普通代码正文更重要。
            if token in chunk["file"].lower():
                score += 2.0

            if token in chunk["name"].lower():
                score += 2.0

    return score


def select_candidates(query, all_chunks):
    """
    先做廉价词法筛选，再调用 embedding-3。
    这样 Rich 不会再把 3454 个 chunk 全部发送给云端模型。
    """
    scored = [
        (_lexical_score(query, chunk), index, chunk)
        for index, chunk in enumerate(all_chunks)
    ]

    scored.sort(key=lambda item: item[0], reverse=True)

    # 有命中的情况下优先保留命中结果。
    selected = [item[2] for item in scored[:MAX_CANDIDATES]]

    return selected


def embed_texts(texts):
    """调用智谱 Embedding-3。"""
    if not texts:
        return []

    vectors = []

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )

        vectors.extend(item.embedding for item in response.data)

    return vectors


def build_index(all_chunks, query):
    """
    不再全量 embedding。
    先筛候选，再使用智谱 embedding-3。
    """
    candidate_chunks = select_candidates(query, all_chunks)

    if not candidate_chunks:
        raise RuntimeError("没有找到可用于代码检索的候选代码块。")

    texts = [
        (
            f'文件: {chunk["file"]}\n'
            f'类型: {chunk["type"]}\n'
            f'名称: {chunk["name"]}\n'
            f'代码:\n{chunk["code"]}'
        )
        for chunk in candidate_chunks
    ]

    try:
        embeddings = embed_texts(texts)
    except Exception as e:
        raise RuntimeError(
            "智谱 Embedding-3 调用失败。\n"
            f"候选 chunk 数量: {len(candidate_chunks)}\n"
            f"原始错误: {e}"
        ) from e

    if not embeddings:
        raise RuntimeError("Embedding-3 没有返回向量。")

    vectors = np.asarray(embeddings, dtype="float32")

    # 不再写死 1024，自动适配 embedding-3 的实际维度。
    dimension = vectors.shape[1]

    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    return index, candidate_chunks


def retrieve_code(query, root="."):
    """
    对外提供的代码检索入口。

    注意：这里每次按照当前 query 做候选筛选，
    避免建立整个仓库的超大 embedding 索引。

    Embedding 服务不可用（余额不足 / 网络错误）时，
    自动降级为纯词法检索，保证 Agent 始终能拿到代码上下文。
    """
    files = collect_python_files(root)

    all_chunks = []

    for file in files:
        try:
            content = read_code_file(file)
            all_chunks.extend(split_code(content, file))
        except (UnicodeDecodeError, OSError):
            continue

    if not all_chunks:
        return []

    try:
        index, indexed_chunks = build_index(all_chunks, query)

        query_embedding = embed_texts([query])[0]
        query_vector = np.asarray([query_embedding], dtype="float32")

        faiss.normalize_L2(query_vector)

        k = min(5, len(indexed_chunks))
        scores, indices = index.search(query_vector, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            results.append(
                {
                    "score": float(score),
                    "file": indexed_chunks[idx]["file"],
                    "type": indexed_chunks[idx]["type"],
                    "name": indexed_chunks[idx]["name"],
                    "code": indexed_chunks[idx]["code"],
                }
            )

        return results

    except Exception as e:
        # ------------------------------------------------
        # Embedding 服务不可用：降级为词法检索。
        # select_candidates 已按词法相关性降序排序。
        # ------------------------------------------------

        print(
            f"⚠️ Embedding 服务不可用（{e}），"
            "已降级为关键词检索。"
        )

        candidates = select_candidates(query, all_chunks)

        results = []

        for chunk in candidates[:5]:
            score = _lexical_score(query, chunk)

            if score <= 0:
                continue

            results.append(
                {
                    "score": score,
                    "file": chunk["file"],
                    "type": chunk["type"],
                    "name": chunk["name"],
                    "code": chunk["code"],
                }
            )

        if not results:
            raise RuntimeError(
                "关键词检索也没有找到相关代码块，"
                f"请尝试更换查询词。原始错误：{e}"
            ) from e

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
    """
    保留旧接口，供已有代码兼容。
    真正的检索推荐使用 retrieve_code(query, root)。
    """
    files = collect_python_files(root)
    all_chunks = []

    for file in files:
        try:
            content = read_code_file(file)
            all_chunks.extend(split_code(content, file))
        except (UnicodeDecodeError, OSError):
            continue

    return all_chunks