"""
test_fix_translator.py - Production-Grade SV-COMP C -> Python QF-LIA Translator.
Fixes:
1. Replaces \t with 4 spaces.
2. C integer division `/` -> Python integer division `//`.
3. Replaces `&&` -> `and`, `||` -> `or`.
4. Replaces comma-separated assignments `i = 0, sum = 0;` -> `i = 0\nsum = 0`.
5. Replaces chained assignments `i = j = k = 0;` -> single assignments.
6. Replaces SV-COMP nondet functions `__VERIFIER_nondet_*` with symbolic arguments.
7. Ensures all blocks have `pass` if empty or comment-only.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import List, Set, Tuple

INDENT = "    "


def _strip_comments(src: str) -> str:
    # First replace tabs with spaces
    src = src.replace("\t", "    ")
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    src = re.sub(r"//[^\n]*", "", src)
    return src


def _extract_globals(src: str) -> List[str]:
    """Extract global constants or definitions outside main()."""
    lines: List[str] = []
    # #define SIZE 100000
    for m in re.finditer(r"#define\s+([a-zA-Z_]\w*)\s+([0-9]+)", src):
        lines.append(f"{m.group(1)} = {m.group(2)}")
    # Global int assignments e.g. int SIZE = 20000001;
    for m in re.finditer(r"\b(?:int|unsigned\s+int|long|short)\s+([a-zA-Z_]\w*)\s*=\s*([0-9]+)\s*;", src):
        main_pos = src.find("main")
        if main_pos == -1 or m.start() < main_pos:
            lines.append(f"{m.group(1)} = {m.group(2)}")
    return lines


def _find_main_body(src: str) -> str:
    src = _strip_comments(src)
    m = re.search(r"int\s+main\s*\([^)]*\)\s*\{", src)
    if not m:
        return ""
    body_start = m.end()
    depth = 1
    i = body_start
    while i < len(src) and depth > 0:
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return src[body_start : i - 1]


def _translate_expression(expr: str) -> str:
    expr = " ".join(expr.split())
    expr = expr.strip()
    expr = expr.replace(";", "")
    # Logical operators (must not use \b because & and | are non-word chars)
    expr = expr.replace("&&", " and ").replace("||", " or ")
    # ! operator (not preceded by alphanumeric or followed by =)
    expr = re.sub(r"(?<![a-zA-Z0-9_])!([^=])", r" not \1", expr)
    # C integer division / -> Python //
    expr = re.sub(r"(?<!/)/(?!/)", "//", expr)
    # SV-COMP nondet functions (all types)
    expr = re.sub(r"__VERIFIER_nondet_(?:uint|int|uchar|ushort|char|bool|short|long)\s*\(\s*\)", "0", expr)
    # Hex literals
    expr = re.sub(r"0[xX]([0-9A-Fa-f]+)", lambda m: str(int(m.group(0), 16)), expr)
    # C suffixes
    expr = re.sub(r"([0-9]+)[uUlL]+", r"\1", expr)
    return expr.strip()


def _compile(expr: str) -> str:
    return _translate_expression(expr)


def _collect_declared(body: str) -> Set[str]:
    declared: Set[str] = set()
    for vm in re.finditer(
        r"\b(?:int|unsigned\s+int|unsigned\s+char|unsigned\s+short|char|long|unsigned\s+long|short)\s+([^;]+?);",
        body,
    ):
        names = vm.group(1)
        for part in names.split(","):
            nm = part.strip().split("=")[0].strip()
            if nm and re.match(r"^[a-zA-Z_]\w*$", nm):
                declared.add(nm)
    return declared


def _extract_block_at(body: str, brace_pos: int) -> Tuple[int, str]:
    assert body[brace_pos] == "{"
    depth = 1
    j = brace_pos + 1
    while j < len(body) and depth > 0:
        if body[j] == "{":
            depth += 1
        elif body[j] == "}":
            depth -= 1
        j += 1
    return j, body[brace_pos + 1 : j - 1]


def _find_matching_paren(body: str, start: int) -> int:
    assert body[start] == "("
    depth = 1
    j = start + 1
    while j < len(body) and depth > 0:
        if body[j] == "(":
            depth += 1
        elif body[j] == ")":
            depth -= 1
        j += 1
    return j - 1


def _find_statement_end(body: str, start: int) -> int:
    i = start
    while i < len(body) and body[i].isspace():
        i += 1
    if i >= len(body):
        return len(body)
    m = re.match(r"(?:for|while|if)\s*\(", body[i:])
    if m:
        open_p = i + m.end() - 1
        close_p = _find_matching_paren(body, open_p)
        after = close_p + 1
        while after < len(body) and body[after].isspace():
            after += 1
        if after < len(body) and body[after] == "{":
            end_brace, _ = _extract_block_at(body, after)
            return end_brace
        else:
            return _find_statement_end(body, after)
    else:
        semi = body.find(";", i)
        return semi + 1 if semi != -1 else len(body)


def _compile_block(body: str, depth: int = 1, params: Set[str] = None) -> List[str]:
    if params is None:
        params = set()
    indent = INDENT * depth
    out: List[str] = []
    i = 0
    n = len(body)

    while i < n:
        # Skip whitespace
        while i < n and body[i].isspace():
            i += 1
        if i >= n:
            break

        # 1. while (cond) { ... } or while (cond) stmt;
        m_while = re.match(r"while\s*\(", body[i:])
        if m_while:
            open_p = i + m_while.end() - 1
            close_p = _find_matching_paren(body, open_p)
            cond = _compile(body[open_p + 1 : close_p])
            after_cond = close_p + 1
            while after_cond < n and body[after_cond].isspace():
                after_cond += 1

            out.append(f"{indent}while {cond}:")
            if after_cond < n and body[after_cond] == "{":
                end_brace, inner = _extract_block_at(body, after_cond)
                inner_stmts = _compile_block(inner, depth + 1)
                valid_stmts = [s for s in inner_stmts if not s.strip().startswith("#")]
                if not valid_stmts:
                    inner_stmts.append(f"{indent}    pass")
                out.extend(inner_stmts)
                i = end_brace
            else:
                stmt_end = _find_statement_end(body, after_cond)
                stmt_text = body[after_cond:stmt_end]
                stmt_compiled = _compile_block(stmt_text, depth + 1)
                valid_stmts = [s for s in stmt_compiled if not s.strip().startswith("#")]
                if not valid_stmts:
                    stmt_compiled.append(f"{indent}    pass")
                out.extend(stmt_compiled)
                i = stmt_end
            continue

        # 2. for (init; cond; step) { ... } or for (...) stmt;
        m_for = re.match(r"for\s*\(", body[i:])
        if m_for:
            open_p = i + m_for.end() - 1
            close_p = _find_matching_paren(body, open_p)
            for_header = body[open_p + 1 : close_p]
            parts = for_header.split(";")
            init = _compile(parts[0]).strip() if len(parts) > 0 else ""
            cond = _compile(parts[1]).strip() if len(parts) > 1 else "True"
            step = _compile(parts[2]).strip() if len(parts) > 2 else ""

            if init:
                init = re.sub(r"^(?:int|unsigned\s+int|long|short)\s+", "", init)
                for ipart in init.split(","):
                    ipart = ipart.strip()
                    if ipart:
                        out.append(f"{indent}{ipart}")

            out.append(f"{indent}while {cond}:")
            after_cond = close_p + 1
            while after_cond < n and body[after_cond].isspace():
                after_cond += 1

            if step:
                m_inc = re.match(r"\+\+([a-zA-Z_]\w*)", step)
                m_inc2 = re.match(r"([a-zA-Z_]\w*)\+\+", step)
                m_dec = re.match(r"--([a-zA-Z_]\w*)", step)
                m_dec2 = re.match(r"([a-zA-Z_]\w*)--", step)
                if m_inc: step = f"{m_inc.group(1)} += 1"
                elif m_inc2: step = f"{m_inc2.group(1)} += 1"
                elif m_dec: step = f"{m_dec.group(1)} -= 1"
                elif m_dec2: step = f"{m_dec2.group(1)} -= 1"

            if after_cond < n and body[after_cond] == "{":
                end_brace, inner = _extract_block_at(body, after_cond)
                inner_stmts = _compile_block(inner, depth + 1)
                if step:
                    inner_stmts.append(f"{indent}    {step}  # for-step")
                valid_stmts = [s for s in inner_stmts if not s.strip().startswith("#")]
                if not valid_stmts:
                    inner_stmts.append(f"{indent}    pass")
                out.extend(inner_stmts)
                i = end_brace
            else:
                stmt_end = _find_statement_end(body, after_cond)
                stmt_text = body[after_cond:stmt_end]
                inner_stmts = _compile_block(stmt_text, depth + 1)
                if step:
                    inner_stmts.append(f"{indent}    {step}  # for-step")
                valid_stmts = [s for s in inner_stmts if not s.strip().startswith("#")]
                if not valid_stmts:
                    inner_stmts.append(f"{indent}    pass")
                out.extend(inner_stmts)
                i = stmt_end
            continue

        # 3. if (cond)
        m_if = re.match(r"if\s*\(", body[i:])
        if m_if:
            open_p = i + m_if.end() - 1
            close_p = _find_matching_paren(body, open_p)
            cond = _compile(body[open_p + 1 : close_p])
            after_cond = close_p + 1
            while after_cond < n and body[after_cond].isspace():
                after_cond += 1

            if after_cond < n and body[after_cond] == "{":
                end_brace, inner = _extract_block_at(body, after_cond)
                if "reach_error" in inner or "abort" in inner:
                    if cond.startswith("not (") and cond.endswith(")"):
                        out.append(f"{indent}assert {cond[5:-1]}")
                    elif cond.startswith("not "):
                        out.append(f"{indent}assert {cond[4:]}")
                    else:
                        out.append(f"{indent}assert not ({cond})")
                    i = end_brace
                    continue

                out.append(f"{indent}if {cond}:")
                inner_stmts = _compile_block(inner, depth + 1)
                valid_stmts = [s for s in inner_stmts if not s.strip().startswith("#")]
                if not valid_stmts:
                    inner_stmts.append(f"{indent}    pass")
                out.extend(inner_stmts)
                i = end_brace
            else:
                stmt_end = _find_statement_end(body, after_cond)
                stmt_text = body[after_cond:stmt_end].strip()
                if "reach_error" in stmt_text or "abort" in stmt_text:
                    if cond.startswith("not (") and cond.endswith(")"):
                        out.append(f"{indent}assert {cond[5:-1]}")
                    elif cond.startswith("not "):
                        out.append(f"{indent}assert {cond[4:]}")
                    else:
                        out.append(f"{indent}assert not ({cond})")
                    i = stmt_end
                    continue

                out.append(f"{indent}if {cond}:")
                inner_stmts = _compile_block(stmt_text, depth + 1)
                if not [s for s in inner_stmts if not s.strip().startswith("#")]:
                    inner_stmts.append(f"{indent}    pass")
                out.extend(inner_stmts)
                i = stmt_end
            continue

        # 3b. else if (cond) -> elif cond:
        m_elif = re.match(r"else\s+if\s*\(", body[i:])
        if m_elif:
            open_p = i + m_elif.end() - 1
            close_p = _find_matching_paren(body, open_p)
            cond = _compile(body[open_p + 1 : close_p])
            after_cond = close_p + 1
            while after_cond < n and body[after_cond].isspace():
                after_cond += 1

            out.append(f"{indent}elif {cond}:")
            if after_cond < n and body[after_cond] == "{":
                end_brace, inner = _extract_block_at(body, after_cond)
                inner_stmts = _compile_block(inner, depth + 1)
                if not [s for s in inner_stmts if not s.strip().startswith("#")]:
                    inner_stmts.append(f"{indent}    pass")
                out.extend(inner_stmts)
                i = end_brace
            else:
                stmt_end = _find_statement_end(body, after_cond)
                stmt_text = body[after_cond:stmt_end].strip()
                inner_stmts = _compile_block(stmt_text, depth + 1)
                if not [s for s in inner_stmts if not s.strip().startswith("#")]:
                    inner_stmts.append(f"{indent}    pass")
                out.extend(inner_stmts)
                i = stmt_end
            continue

        # 3c. else -> else:
        m_else = re.match(r"else\b", body[i:])
        if m_else:
            after_else = i + m_else.end()
            while after_else < n and body[after_else].isspace():
                after_else += 1
            out.append(f"{indent}else:")
            if after_else < n and body[after_else] == "{":
                end_brace, inner = _extract_block_at(body, after_else)
                inner_stmts = _compile_block(inner, depth + 1)
                if not [s for s in inner_stmts if not s.strip().startswith("#")]:
                    inner_stmts.append(f"{indent}    pass")
                out.extend(inner_stmts)
                i = end_brace
            else:
                stmt_end = _find_statement_end(body, after_else)
                stmt_text = body[after_else:stmt_end].strip()
                inner_stmts = _compile_block(stmt_text, depth + 1)
                if not [s for s in inner_stmts if not s.strip().startswith("#")]:
                    inner_stmts.append(f"{indent}    pass")
                out.extend(inner_stmts)
                i = stmt_end
            continue

        # 4. __VERIFIER_assert(cond);
        m_assert = re.match(r"__VERIFIER_assert\s*\(", body[i:])
        if m_assert:
            open_p = i + m_assert.end() - 1
            close_p = _find_matching_paren(body, open_p)
            cond = _compile(body[open_p + 1 : close_p])
            semi = body.find(";", close_p)
            out.append(f"{indent}assert {cond}")
            i = semi + 1 if semi != -1 else close_p + 1
            continue

        # 5. reach_error(); abort();
        m_reach = re.match(r"(?:reach_error\s*\(\s*\)\s*;\s*)?abort\s*\(\s*\)\s*;", body[i:])
        if m_reach:
            out.append(f"{indent}assert False, 'reach_error'")
            i += m_reach.end()
            continue

        # 6. assume_abort_if_not(cond);
        m_assume = re.match(r"assume_abort_if_not\s*\(", body[i:])
        if m_assume:
            open_p = i + m_assume.end() - 1
            close_p = _find_matching_paren(body, open_p)
            cond = _compile(body[open_p + 1 : close_p])
            semi = body.find(";", close_p)
            out.append(f"{indent}if not ({cond}): return")
            i = semi + 1 if semi != -1 else close_p + 1
            continue

        # 7. return ...;
        m_ret = re.match(r"return\s*([^;]*?)\s*;", body[i:])
        if m_ret:
            ret_val = _compile(m_ret.group(1)).strip()
            if ret_val and ret_val != "0":
                out.append(f"{indent}return {ret_val}")
            else:
                out.append(f"{indent}return")
            i += m_ret.end()
            continue

        # 8. Increment / Decrement: ++x, x++, --x, x--
        m_inc = re.match(r"\+\+([a-zA-Z_]\w*)\s*;", body[i:])
        if m_inc:
            out.append(f"{indent}{m_inc.group(1)} += 1")
            i += m_inc.end()
            continue
        m_inc2 = re.match(r"([a-zA-Z_]\w*)\+\+\s*;", body[i:])
        if m_inc2:
            out.append(f"{indent}{m_inc2.group(1)} += 1")
            i += m_inc2.end()
            continue
        m_dec = re.match(r"--([a-zA-Z_]\w*)\s*;", body[i:])
        if m_dec:
            out.append(f"{indent}{m_dec.group(1)} -= 1")
            i += m_dec.end()
            continue
        m_dec2 = re.match(r"([a-zA-Z_]\w*)--\s*;", body[i:])
        if m_dec2:
            out.append(f"{indent}{m_dec2.group(1)} -= 1")
            i += m_dec2.end()
            continue

        # 9. Augmented assignment: x += expr; x -= expr; etc.
        m_aug = re.match(r"([a-zA-Z_]\w*)\s*(\+=|-=|\*=|//=|%=|&=|\|=|\^=)\s*([^;]+?)\s*;", body[i:])
        if m_aug:
            var = m_aug.group(1)
            op = m_aug.group(2)
            val = _compile(m_aug.group(3))
            out.append(f"{indent}{var} {op} {val}")
            i += m_aug.end()
            continue

        # 10. Chained assignments: i = j = k = 0; -> k = 0, j = k, i = j
        m_chain = re.match(r"((?:[a-zA-Z_]\w*\s*=\s*){2,})([^;]+?)\s*;", body[i:])
        if m_chain:
            targets = [t.strip() for t in m_chain.group(1).split("=") if t.strip()]
            val = _compile(m_chain.group(2))
            out.append(f"{indent}{targets[-1]} = {val}")
            for t_idx in range(len(targets) - 2, -1, -1):
                out.append(f"{indent}{targets[t_idx]} = {targets[t_idx + 1]}")
            i += m_chain.end()
            continue

        # 11. Comma-separated assignments: i = 0, sum = 0;
        m_comma_assign = re.match(r"([a-zA-Z_]\w*\s*=[^,;]+(?:\s*,\s*[a-zA-Z_]\w*\s*=[^,;]+)+)\s*;", body[i:])
        if m_comma_assign:
            parts = m_comma_assign.group(1).split(",")
            for p in parts:
                p = p.strip()
                if "=" in p:
                    v_name, v_val = p.split("=", 1)
                    v_name = v_name.strip()
                    v_val = _compile(v_val)
                    if v_name:
                        out.append(f"{indent}{v_name} = {v_val}")
            i += m_comma_assign.end()
            continue

        # 12. Multi-declaration with assignments: int x = 0, y = 50000, z = 0;
        m_multi_decl = re.match(r"(?:int|unsigned\s+int|long|unsigned\s+char|char|short)\s+([^;]+?);", body[i:])
        if m_multi_decl:
            decl_content = m_multi_decl.group(1)
            parts = decl_content.split(",")
            for p in parts:
                p = p.strip()
                if "=" in p:
                    v_name, v_val = p.split("=", 1)
                    v_name = v_name.strip()
                    if v_name in params:
                        continue
                    val = _compile(v_val)
                    if v_name:
                        out.append(f"{indent}{v_name} = {val}")
            i += m_multi_decl.end()
            continue

        # 13. Single-target assignment: var = expr;
        m_assign = re.match(r"([a-zA-Z_]\w*)\s*=\s*([^;]+?)\s*;", body[i:])
        if m_assign:
            var = m_assign.group(1)
            raw_val = m_assign.group(2).strip()
            if var in params and ("nondet" in raw_val or raw_val == "0"):
                i += m_assign.end()
                continue
            val = _compile(raw_val)
            out.append(f"{indent}{var} = {val}")
            i += m_assign.end()
            continue

        # 14. Skip lone semicolons or braces
        if body[i] in (";", "{", "}"):
            i += 1
            continue

        # Unknown statement
        semi = body.find(";", i)
        if semi != -1:
            raw = body[i:semi].strip()
            if raw:
                out.append(f"{indent}# {raw}")
            i = semi + 1
        else:
            i += 1

    return out


def translate_c_code(c_src: str) -> str:
    c_src = _strip_comments(c_src)
    body = _find_main_body(c_src)
    if not body:
        raise ValueError("Could not find main() body")

    declared = _collect_declared(body)
    globals_list = _extract_globals(c_src)

    # Parameters: variables initialized to __VERIFIER_nondet or uninitialized
    params: List[str] = []
    for var in sorted(declared):
        if re.search(r"\b" + var + r"\s*=\s*__VERIFIER_nondet", body) or not re.search(r"\b" + var + r"\s*=", body):
            params.append(var)

    param_str = ", ".join(params)
    compiled_stmts = _compile_block(body, depth=1, params=set(params))

    all_stmts = []
    for g in globals_list:
        all_stmts.append(f"{INDENT}{g}")
    all_stmts.extend(compiled_stmts)

    valid_stmts = [s for s in all_stmts if not s.strip().startswith("#")]
    if not valid_stmts:
        all_stmts.append(f"{INDENT}pass")

    code = f"def f({param_str}):\n"
    code += "\n".join(all_stmts)
    if not code.endswith("\n"):
        code += "\n"
    return code


def translate_file(c_path: Path, py_path: Path) -> str:
    c_src = c_path.read_text(encoding="utf-8", errors="ignore")
    code = translate_c_code(c_src)
    py_path.parent.mkdir(parents=True, exist_ok=True)
    py_path.write_text(code, encoding="utf-8")
    return code

