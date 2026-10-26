"""Best-effort, dependency-free symbol and import extraction.

This is intentionally *not* a compiler front end.  The goal is a searchable,
explainable map of a repository: names, locations, imports and short doc
snippets.  Python uses :mod:`ast`; other languages use conservative line
patterns, and every extraction is marked with the line it came from so a
human can check it.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from .models import ImportRef, Symbol
from .util import first_sentence, split_lines, truncate

MAX_SYMBOLS = 400
MAX_IMPORTS = 150

JS_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "function",
    "constructor",
    "super",
    "await",
    "else",
    "do",
    "try",
    "typeof",
    "new",
}


@dataclass
class ParseResult:
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[ImportRef] = field(default_factory=list)
    summary: str = ""


def parse_file(lang: str, text: str, path: str = "") -> ParseResult:
    parser = {
        "python": _parse_python,
        "javascript": _parse_js,
        "typescript": _parse_js,
        "vue": _parse_js,
        "svelte": _parse_js,
        "go": _parse_go,
        "rust": _parse_rust,
        "java": _parse_c_family,
        "kotlin": _parse_c_family,
        "scala": _parse_c_family,
        "csharp": _parse_c_family,
        "c": _parse_c_family,
        "cpp": _parse_c_family,
        "objective-c": _parse_c_family,
        "swift": _parse_swift,
        "ruby": _parse_ruby,
        "php": _parse_php,
        "shell": _parse_shell,
        "powershell": _parse_shell,
        "elixir": _parse_elixir,
        "haskell": _parse_haskell,
        "markdown": _parse_markdown,
        "rst": _parse_rst,
        "html": _parse_html,
        "lua": _parse_lua,
        "perl": _parse_perl,
        "r": _parse_generic_functions,
        "julia": _parse_generic_functions,
        "dart": _parse_c_family,
    }.get(lang)
    if parser is None:
        return ParseResult()
    try:
        result = parser(text)
    except (RecursionError, ValueError, re.error):
        return ParseResult()
    result.symbols = _dedupe(result.symbols)[:MAX_SYMBOLS]
    result.imports = result.imports[:MAX_IMPORTS]
    result.summary = truncate(result.summary, 240)
    return result


def _dedupe(symbols: list[Symbol]) -> list[Symbol]:
    seen: set[tuple[str, int, str]] = set()
    output: list[Symbol] = []
    for symbol in symbols:
        key = (symbol.name, symbol.line, symbol.kind)
        if key in seen:
            continue
        seen.add(key)
        output.append(symbol)
    return output


def _leading_doc(lines: list[str], line: int) -> str:
    """Collect a JS/Ruby-style comment block immediately above ``line``."""
    index = line - 2
    collected: list[str] = []
    while index >= 0 and len(collected) < 6:
        stripped = lines[index].strip()
        if stripped.startswith(("//", "#", "*", "/*", "///", "//!")):
            cleaned = stripped.lstrip("/*#! ").strip()
            if cleaned and not cleaned.startswith("@"):
                collected.append(cleaned)
            index -= 1
            continue
        break
    if not collected:
        return ""
    return first_sentence(" ".join(reversed(collected)), 160)


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


def _parse_python(text: str) -> ParseResult:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return _parse_python_regex(text)

    symbols: list[Symbol] = []
    imports: list[ImportRef] = []
    stack: list[tuple[str, str]] = []  # (name, kind)

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            container = ".".join(name for name, _ in stack)
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(base.attr)
            signature = "class " + node.name + (f"({', '.join(bases)})" if bases else "")
            symbols.append(
                Symbol(
                    name=node.name,
                    kind="class",
                    line=node.lineno,
                    end_line=getattr(node, "end_lineno", None),
                    signature=truncate(signature, 120),
                    doc=_doc(node),
                    exported=not node.name.startswith("_"),
                    container=container,
                )
            )
            stack.append((node.name, "class"))
            self.generic_visit(node)
            stack.pop()

        def _function(self, node: ast.AST, kind: str) -> None:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return
            container = ".".join(name for name, _ in stack)
            immediate = stack[-1][1] if stack else ""
            actual_kind = "method" if immediate == "class" else "function"
            symbols.append(
                Symbol(
                    name=node.name,
                    kind=actual_kind,
                    line=node.lineno,
                    end_line=getattr(node, "end_lineno", None),
                    signature=truncate(_py_signature(node), 160),
                    doc=_doc(node),
                    exported=not node.name.startswith("_"),
                    container=container,
                )
            )
            stack.append((node.name, "function"))
            self.generic_visit(node)
            stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._function(node, "function")

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._function(node, "function")

        def visit_Assign(self, node: ast.Assign) -> None:
            if not stack:
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        symbols.append(
                            Symbol(
                                name=target.id,
                                kind="constant",
                                line=node.lineno,
                                signature=truncate(ast.get_source_segment(text, node) or target.id, 120),
                                exported=not target.id.startswith("_"),
                            )
                        )
            self.generic_visit(node)

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                imports.append(ImportRef(module=alias.name, line=node.lineno, raw=f"import {alias.name}"))

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            module = "." * (node.level or 0) + (node.module or "")
            imports.append(
                ImportRef(
                    module=module,
                    line=node.lineno,
                    raw=f"from {module} import {', '.join(alias.name for alias in node.names)}",
                )
            )

    Visitor().visit(tree)
    summary = _doc(tree)
    return ParseResult(symbols=symbols, imports=imports, summary=summary)


def _doc(node: ast.AST) -> str:
    try:
        value = ast.get_docstring(node, clean=True)
    except TypeError:
        return ""
    if not value:
        return ""
    return first_sentence(value, 180)


def _py_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args: list[str] = []
    for arg in [*node.args.posonlyargs, *node.args.args]:
        args.append(arg.arg)
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    for arg in node.args.kwonlyargs:
        args.append(arg.arg)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(args)})"


_PY_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)")
_PY_CLASS_RE = re.compile(r"^\s*class\s+(\w+)(?:\(([^)]*)\))?")
_PY_IMPORT_RE = re.compile(r"^\s*from\s+([\w.]+)\s+import\s+(.+)$|^\s*import\s+([\w., ]+)$")


def _parse_python_regex(text: str) -> ParseResult:
    """Fallback for files with syntax errors (e.g. Python 2 sources)."""
    lines = split_lines(text)
    symbols: list[Symbol] = []
    imports: list[ImportRef] = []
    container = ""
    for number, line in enumerate(lines, start=1):
        class_match = _PY_CLASS_RE.match(line)
        def_match = _PY_DEF_RE.match(line)
        if class_match:
            container = class_match.group(1)
            symbols.append(
                Symbol(
                    name=container,
                    kind="class",
                    line=number,
                    signature=truncate(line.strip(), 120),
                    doc=_leading_doc(lines, number),
                    exported=not container.startswith("_"),
                )
            )
        elif def_match:
            name = def_match.group(1)
            symbols.append(
                Symbol(
                    name=name,
                    kind="method" if container else "function",
                    line=number,
                    signature=truncate(line.strip(), 160),
                    doc=_leading_doc(lines, number),
                    exported=not name.startswith("_"),
                    container=container,
                )
            )
        elif line and not line[0].isspace():
            container = ""
        import_match = _PY_IMPORT_RE.match(line)
        if import_match:
            module = import_match.group(1) or (import_match.group(3) or "").strip()
            imports.append(ImportRef(module=module, line=number, raw=line.strip()))
    summary = first_sentence("\n".join(line for line in lines[:20] if line.strip()), 200)
    return ParseResult(symbols=symbols, imports=imports, summary=summary)


# ---------------------------------------------------------------------------
# JavaScript / TypeScript
# ---------------------------------------------------------------------------

_JS_CLASS_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?(class)\s+([A-Za-z_$][\w$]*)"
)
_JS_FUNCTION_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*\("
)
_JS_VAR_FN_RE = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::\s*[^=]+)?=\s*"
    r"(?:async\s+)?(?:function\b|(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=>)"
)
_JS_TYPE_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(interface|type|enum|namespace)\s+([A-Za-z_$][\w$]*)"
)
_JS_METHOD_RE = re.compile(
    r"^\s{2,}(?:(?:public|private|protected|static|async|get|set|readonly|override|abstract)\s+)*"
    r"([A-Za-z_$][\w$]*)\s*\(([^)]*)\)\s*(?::\s*[^{;]+)?\{\s*$"
)
_JS_IMPORT_RE = re.compile(
    r"^\s*import\s+(?:type\s+)?(?:[\w$*{}\s,]+?\s+from\s+)?[\"']([^\"']+)[\"']"
)
_JS_EXPORT_FROM_RE = re.compile(r"^\s*export\s+(?:type\s+)?(?:\*|\{[^}]*\})\s+from\s+[\"']([^\"']+)[\"']")
_JS_REQUIRE_RE = re.compile(r"(?:require|import)\(\s*[\"']([^\"']+)[\"']\s*\)")


def _brace_delta(line: str) -> int:
    cleaned = re.sub(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`", "", line)
    cleaned = cleaned.split("//", 1)[0]
    return cleaned.count("{") - cleaned.count("}")


def _js_class_ranges(lines: list[str]) -> list[tuple[int, int, str]]:
    ranges: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        match = _JS_CLASS_RE.match(line)
        if not match:
            continue
        name = match.group(2)
        depth = 0
        started = False
        end = len(lines) - 1
        for cursor in range(index, min(len(lines), index + 4000)):
            delta = _brace_delta(lines[cursor])
            if delta:
                started = True
            depth += delta
            if started and depth <= 0:
                end = cursor
                break
        ranges.append((index, end, name))
    return ranges


def _parse_js(text: str) -> ParseResult:
    lines = split_lines(text)
    symbols: list[Symbol] = []
    imports: list[ImportRef] = []
    ranges = _js_class_ranges(lines)

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "/*", "*")):
            continue

        class_match = _JS_CLASS_RE.match(line)
        if class_match:
            name = class_match.group(2)
            symbols.append(
                Symbol(
                    name=name,
                    kind="class",
                    line=index + 1,
                    signature=truncate(stripped, 140),
                    doc=_leading_doc(lines, index + 1),
                    exported="export" in line,
                )
            )
            continue

        function_match = _JS_FUNCTION_RE.match(line)
        if function_match:
            name = function_match.group(1)
            symbols.append(
                Symbol(
                    name=name,
                    kind="function",
                    line=index + 1,
                    signature=truncate(stripped, 160),
                    doc=_leading_doc(lines, index + 1),
                    exported="export" in line,
                )
            )

        var_match = _JS_VAR_FN_RE.match(line)
        if var_match:
            name = var_match.group(1)
            symbols.append(
                Symbol(
                    name=name,
                    kind="function",
                    line=index + 1,
                    signature=truncate(stripped, 160),
                    doc=_leading_doc(lines, index + 1),
                    exported="export" in line,
                )
            )

        type_match = _JS_TYPE_RE.match(line)
        if type_match:
            symbols.append(
                Symbol(
                    name=type_match.group(2),
                    kind=type_match.group(1),
                    line=index + 1,
                    signature=truncate(stripped, 140),
                    doc=_leading_doc(lines, index + 1),
                    exported="export" in line,
                )
            )

        for match in _JS_IMPORT_RE.finditer(line):
            imports.append(ImportRef(module=match.group(1), line=index + 1, raw=stripped))
        for match in _JS_EXPORT_FROM_RE.finditer(line):
            imports.append(ImportRef(module=match.group(1), line=index + 1, raw=stripped))
        for match in _JS_REQUIRE_RE.finditer(line):
            imports.append(ImportRef(module=match.group(1), line=index + 1, raw=stripped))

    for start, end, class_name in ranges:
        for index in range(start + 1, min(end, start + 500) + 1):
            if index >= len(lines):
                break
            method_match = _JS_METHOD_RE.match(lines[index])
            if not method_match:
                continue
            name = method_match.group(1)
            if name in JS_KEYWORDS:
                continue
            symbols.append(
                Symbol(
                    name=name,
                    kind="method",
                    line=index + 1,
                    signature=truncate(lines[index].strip(), 160),
                    doc=_leading_doc(lines, index + 1),
                    exported=True,
                    container=class_name,
                )
            )

    summary = _leading_doc(lines, 1)
    if not summary:
        for line in lines[:30]:
            stripped = line.strip()
            if stripped.startswith(("//", "/*", "*", "#!")):
                summary = stripped.lstrip("/*#! ").strip()
                if summary:
                    break
    return ParseResult(symbols=symbols, imports=imports, summary=summary)


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------

_GO_FUNC_RE = re.compile(r"^func\s+(?:\(([^)]*)\)\s+)?(\w+)\s*\(")
_GO_TYPE_RE = re.compile(r"^type\s+(\w+)\s+(struct|interface|func|map|\[\]|chan|\w+)")
_GO_IMPORT_RE = re.compile(r"^\s*(?:import\s+)?([\w.-]*)\s*[\"`]([^\"`]+)[\"`]")


def _parse_go(text: str) -> ParseResult:
    lines = split_lines(text)
    symbols: list[Symbol] = []
    imports: list[ImportRef] = []
    in_import_block = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ("):
            in_import_block = True
            continue
        if in_import_block:
            if stripped.startswith(")"):
                in_import_block = False
                continue
            match = _GO_IMPORT_RE.match(line)
            if match:
                imports.append(ImportRef(module=match.group(2), line=index + 1, raw=stripped))
        elif stripped.startswith("import "):
            match = _GO_IMPORT_RE.match(line)
            if match:
                imports.append(ImportRef(module=match.group(2), line=index + 1, raw=stripped))

        func_match = _GO_FUNC_RE.match(line)
        if func_match:
            receiver = func_match.group(1) or ""
            name = func_match.group(2)
            container = ""
            if receiver:
                parts = receiver.replace("*", " ").split()
                container = parts[-1] if parts else ""
            symbols.append(
                Symbol(
                    name=name,
                    kind="method" if receiver else "function",
                    line=index + 1,
                    signature=truncate(stripped, 160),
                    doc=_leading_doc(lines, index + 1),
                    exported=name[:1].isupper(),
                    container=container,
                )
            )
            continue

        type_match = _GO_TYPE_RE.match(line)
        if type_match:
            name = type_match.group(1)
            kind = type_match.group(2)
            symbols.append(
                Symbol(
                    name=name,
                    kind="type" if kind in {"struct", "interface"} else "alias",
                    line=index + 1,
                    signature=truncate(stripped, 140),
                    doc=_leading_doc(lines, index + 1),
                    exported=name[:1].isupper(),
                )
            )

    summary = ""
    for line in lines[:30]:
        if line.startswith("// "):
            summary = line[3:].strip()
            break
    return ParseResult(symbols=symbols, imports=imports, summary=summary)


# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------

_RS_FN_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)")
_RS_TYPE_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(struct|enum|trait|type|union)\s+(\w+)"
)
_RS_IMPL_RE = re.compile(r"^\s*impl(?:<[^>]*>)?\s+(?:[\w:<>]+\s+for\s+)?([\w:<>]+)")
_RS_MOD_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)")
_RS_USE_RE = re.compile(r"^\s*(?:pub\s+)?use\s+([^;]+);")


def _parse_rust(text: str) -> ParseResult:
    lines = split_lines(text)
    symbols: list[Symbol] = []
    imports: list[ImportRef] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        fn_match = _RS_FN_RE.match(line)
        if fn_match:
            name = fn_match.group(1)
            symbols.append(
                Symbol(
                    name=name,
                    kind="function",
                    line=index + 1,
                    signature=truncate(stripped, 160),
                    doc=_leading_doc(lines, index + 1),
                    exported="pub" in stripped,
                )
            )
        type_match = _RS_TYPE_RE.match(line)
        if type_match:
            symbols.append(
                Symbol(
                    name=type_match.group(2),
                    kind=type_match.group(1),
                    line=index + 1,
                    signature=truncate(stripped, 140),
                    doc=_leading_doc(lines, index + 1),
                    exported="pub" in stripped,
                )
            )
        impl_match = _RS_IMPL_RE.match(line)
        if impl_match:
            target = impl_match.group(1).split("<", 1)[0].split("::")[-1]
            symbols.append(
                Symbol(
                    name=target,
                    kind="impl",
                    line=index + 1,
                    signature=truncate(stripped, 140),
                    doc=_leading_doc(lines, index + 1),
                )
            )
        mod_match = _RS_MOD_RE.match(line)
        if mod_match and "fn " not in line:
            symbols.append(
                Symbol(
                    name=mod_match.group(1),
                    kind="module",
                    line=index + 1,
                    signature=truncate(stripped, 120),
                    doc=_leading_doc(lines, index + 1),
                )
            )
        use_match = _RS_USE_RE.match(line)
        if use_match:
            module = use_match.group(1).split("{", 1)[0].strip().replace("::", ".")
            imports.append(ImportRef(module=module, line=index + 1, raw=stripped))
    summary = _leading_doc(lines, 1)
    return ParseResult(symbols=symbols, imports=imports, summary=summary)


# ---------------------------------------------------------------------------
# C-family (Java, Kotlin, C#, C, C++, Scala, Dart)
# ---------------------------------------------------------------------------

_CF_TYPE_RE = re.compile(
    r"^\s*(?:(?:public|private|protected|internal|abstract|final|static|sealed|data|open|partial|"
    r"export|declare|unsafe|readonly)\s+)*(class|interface|enum|record|struct|object|protocol|extension|trait)"
    r"\s+([A-Za-z_]\w*)"
)
_CF_METHOD_RE = re.compile(
    r"^\s*(?:(?:public|private|protected|internal|static|final|abstract|virtual|override|synchronized|"
    r"native|async|inline|operator|extern|constexpr|friend|virtual)\s+)*"
    r"(?:[A-Za-z_][\w<>\[\],.?*&:]*\s+)+"
    r"([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:const\s*)?(?:\{|;|=>)"
)
_CF_IMPORT_RE = re.compile(r"^\s*(?:import|using|use)\s+(?:static\s+)?([\w.:*]+)\s*;?")


def _parse_c_family(text: str) -> ParseResult:
    lines = split_lines(text)
    symbols: list[Symbol] = []
    imports: list[ImportRef] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("//", "/*", "*")):
            continue
        type_match = _CF_TYPE_RE.match(line)
        if type_match:
            name = type_match.group(2)
            symbols.append(
                Symbol(
                    name=name,
                    kind=type_match.group(1),
                    line=index + 1,
                    signature=truncate(stripped, 140),
                    doc=_leading_doc(lines, index + 1),
                    exported=not stripped.startswith("private"),
                )
            )
            continue
        method_match = _CF_METHOD_RE.match(line)
        if method_match:
            name = method_match.group(1)
            if name not in JS_KEYWORDS and name not in {"if", "for", "while", "switch", "catch", "sizeof"}:
                symbols.append(
                    Symbol(
                        name=name,
                        kind="function",
                        line=index + 1,
                        signature=truncate(stripped, 160),
                        doc=_leading_doc(lines, index + 1),
                    )
                )
        import_match = _CF_IMPORT_RE.match(line)
        if import_match:
            imports.append(ImportRef(module=import_match.group(1), line=index + 1, raw=stripped))
    summary = _leading_doc(lines, 1)
    return ParseResult(symbols=symbols, imports=imports, summary=summary)


# ---------------------------------------------------------------------------
# Swift
# ---------------------------------------------------------------------------

_SWIFT_TYPE_RE = re.compile(r"^\s*(?:public\s+|private\s+|internal\s+|open\s+|final\s+)*(class|struct|enum|protocol|extension)\s+(\w+)")
_SWIFT_FUNC_RE = re.compile(r"^\s*(?:public\s+|private\s+|internal\s+|open\s+|static\s+|class\s+)?func\s+(\w+)")
_SWIFT_IMPORT_RE = re.compile(r"^\s*import\s+(\w+)")


def _parse_swift(text: str) -> ParseResult:
    lines = split_lines(text)
    symbols: list[Symbol] = []
    imports: list[ImportRef] = []
    for index, line in enumerate(lines):
        type_match = _SWIFT_TYPE_RE.match(line)
        if type_match:
            symbols.append(
                Symbol(
                    name=type_match.group(2),
                    kind=type_match.group(1),
                    line=index + 1,
                    signature=truncate(line.strip(), 140),
                    doc=_leading_doc(lines, index + 1),
                )
            )
        func_match = _SWIFT_FUNC_RE.match(line)
        if func_match:
            symbols.append(
                Symbol(
                    name=func_match.group(1),
                    kind="function",
                    line=index + 1,
                    signature=truncate(line.strip(), 160),
                    doc=_leading_doc(lines, index + 1),
                )
            )
        import_match = _SWIFT_IMPORT_RE.match(line)
        if import_match:
            imports.append(ImportRef(module=import_match.group(1), line=index + 1, raw=line.strip()))
    return ParseResult(symbols=symbols, imports=imports, summary=_leading_doc(lines, 1))


# ---------------------------------------------------------------------------
# Ruby / PHP
# ---------------------------------------------------------------------------

_RB_DEF_RE = re.compile(r"^\s*def\s+(?:self\.)?([\w?!]+)")
_RB_TYPE_RE = re.compile(r"^\s*(class|module)\s+([\w:]+)")
_RB_REQUIRE_RE = re.compile(r"^\s*require(?:_relative)?\s+[\"']([^\"']+)[\"']")


def _parse_ruby(text: str) -> ParseResult:
    lines = split_lines(text)
    symbols: list[Symbol] = []
    imports: list[ImportRef] = []
    container = ""
    for index, line in enumerate(lines):
        type_match = _RB_TYPE_RE.match(line)
        if type_match:
            name = type_match.group(2)
            container = name
            symbols.append(
                Symbol(
                    name=name,
                    kind=type_match.group(1),
                    line=index + 1,
                    signature=truncate(line.strip(), 140),
                    doc=_leading_doc(lines, index + 1),
                )
            )
        def_match = _RB_DEF_RE.match(line)
        if def_match:
            name = def_match.group(1)
            symbols.append(
                Symbol(
                    name=name,
                    kind="method" if container else "function",
                    line=index + 1,
                    signature=truncate(line.strip(), 160),
                    doc=_leading_doc(lines, index + 1),
                    container=container,
                )
            )
        require_match = _RB_REQUIRE_RE.match(line)
        if require_match:
            imports.append(ImportRef(module=require_match.group(1), line=index + 1, raw=line.strip()))
    return ParseResult(symbols=symbols, imports=imports, summary=_leading_doc(lines, 1))


_PHP_FUNC_RE = re.compile(
    r"^\s*(?:(?:public|private|protected|static|final|abstract)\s+)*function\s+(\w+)\s*\("
)
_PHP_TYPE_RE = re.compile(r"^\s*(?:final\s+|abstract\s+)?(class|interface|trait|enum)\s+(\w+)")
_PHP_USE_RE = re.compile(r"^\s*use\s+([\w\\]+)")


def _parse_php(text: str) -> ParseResult:
    lines = split_lines(text)
    symbols: list[Symbol] = []
    imports: list[ImportRef] = []
    for index, line in enumerate(lines):
        type_match = _PHP_TYPE_RE.match(line)
        if type_match:
            symbols.append(
                Symbol(
                    name=type_match.group(2),
                    kind=type_match.group(1),
                    line=index + 1,
                    signature=truncate(line.strip(), 140),
                    doc=_leading_doc(lines, index + 1),
                )
            )
        func_match = _PHP_FUNC_RE.match(line)
        if func_match:
            symbols.append(
                Symbol(
                    name=func_match.group(1),
                    kind="function",
                    line=index + 1,
                    signature=truncate(line.strip(), 160),
                    doc=_leading_doc(lines, index + 1),
                )
            )
        use_match = _PHP_USE_RE.match(line)
        if use_match:
            imports.append(ImportRef(module=use_match.group(1), line=index + 1, raw=line.strip()))
    return ParseResult(symbols=symbols, imports=imports, summary=_leading_doc(lines, 1))


# ---------------------------------------------------------------------------
# Shell / Elixir / Haskell / Lua / Perl / generic
# ---------------------------------------------------------------------------

_SH_FUNC_RE = re.compile(r"^\s*(?:function\s+)?([A-Za-z_][\w-]*)\s*\(\)\s*\{?")
_EX_DEF_RE = re.compile(r"^\s*def(?:p|module|macro)?\s+([\w!?]+)")
_EX_MODULE_RE = re.compile(r"^\s*defmodule\s+([\w.]+)")
_HS_SIG_RE = re.compile(r"^([a-z_][\w']*)\s*::")
_LUA_FUNC_RE = re.compile(r"^\s*(?:local\s+)?function\s+([\w.:]+)")
_PL_SUB_RE = re.compile(r"^\s*sub\s+(\w+)")


def _parse_shell(text: str) -> ParseResult:
    lines = split_lines(text)
    symbols: list[Symbol] = []
    for index, line in enumerate(lines):
        match = _SH_FUNC_RE.match(line)
        if match and match.group(1) not in {"if", "for", "while", "case", "do"}:
            symbols.append(
                Symbol(
                    name=match.group(1),
                    kind="function",
                    line=index + 1,
                    signature=truncate(line.strip(), 120),
                    doc=_leading_doc(lines, index + 1),
                )
            )
    return ParseResult(symbols=symbols, summary=_leading_doc(lines, 1))


def _parse_elixir(text: str) -> ParseResult:
    lines = split_lines(text)
    symbols: list[Symbol] = []
    imports: list[ImportRef] = []
    for index, line in enumerate(lines):
        module_match = _EX_MODULE_RE.match(line)
        if module_match:
            symbols.append(
                Symbol(
                    name=module_match.group(1).split(".")[-1],
                    kind="module",
                    line=index + 1,
                    signature=truncate(line.strip(), 140),
                    container=module_match.group(1),
                )
            )
            continue
        def_match = _EX_DEF_RE.match(line)
        if def_match:
            symbols.append(
                Symbol(
                    name=def_match.group(1),
                    kind="function",
                    line=index + 1,
                    signature=truncate(line.strip(), 160),
                    doc=_leading_doc(lines, index + 1),
                )
            )
        for keyword in ("alias", "import", "require", "use"):
            if line.strip().startswith(keyword + " "):
                module = line.strip().split(None, 1)[1].split(",", 1)[0]
                imports.append(ImportRef(module=module, line=index + 1, raw=line.strip()))
    return ParseResult(symbols=symbols, imports=imports, summary=_leading_doc(lines, 1))


def _parse_haskell(text: str) -> ParseResult:
    lines = split_lines(text)
    symbols: list[Symbol] = []
    imports: list[ImportRef] = []
    for index, line in enumerate(lines):
        if line.startswith("import "):
            module = line.split()[1] if len(line.split()) > 1 else ""
            imports.append(ImportRef(module=module, line=index + 1, raw=line.strip()))
            continue
        match = _HS_SIG_RE.match(line)
        if match:
            symbols.append(
                Symbol(
                    name=match.group(1),
                    kind="function",
                    line=index + 1,
                    signature=truncate(line.strip(), 160),
                    doc=_leading_doc(lines, index + 1),
                )
            )
    return ParseResult(symbols=symbols, imports=imports, summary=_leading_doc(lines, 1))


def _parse_lua(text: str) -> ParseResult:
    lines = split_lines(text)
    symbols: list[Symbol] = []
    for index, line in enumerate(lines):
        match = _LUA_FUNC_RE.match(line)
        if match:
            symbols.append(
                Symbol(
                    name=match.group(1).split(".")[-1],
                    kind="function",
                    line=index + 1,
                    signature=truncate(line.strip(), 160),
                    container=match.group(1).rsplit(".", 1)[0] if "." in match.group(1) else "",
                )
            )
    return ParseResult(symbols=symbols, summary=_leading_doc(lines, 1))


def _parse_perl(text: str) -> ParseResult:
    lines = split_lines(text)
    symbols: list[Symbol] = []
    imports: list[ImportRef] = []
    for index, line in enumerate(lines):
        if line.startswith("use ") or line.startswith("require "):
            module = line.split()[1].rstrip(";") if len(line.split()) > 1 else ""
            imports.append(ImportRef(module=module, line=index + 1, raw=line.strip()))
        match = _PL_SUB_RE.match(line)
        if match:
            symbols.append(
                Symbol(
                    name=match.group(1),
                    kind="function",
                    line=index + 1,
                    signature=truncate(line.strip(), 160),
                )
            )
    return ParseResult(symbols=symbols, imports=imports, summary=_leading_doc(lines, 1))


def _parse_generic_functions(text: str) -> ParseResult:
    lines = split_lines(text)
    symbols: list[Symbol] = []
    for index, line in enumerate(lines):
        match = re.match(r"^\s*(?:function\s+)?([A-Za-z_]\w*)\s*(?:<-|=)\s*function", line)
        if match:
            symbols.append(
                Symbol(
                    name=match.group(1),
                    kind="function",
                    line=index + 1,
                    signature=truncate(line.strip(), 160),
                )
            )
    return ParseResult(symbols=symbols, summary=_leading_doc(lines, 1))


# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------


def _parse_markdown(text: str) -> ParseResult:
    lines = split_lines(text)
    symbols: list[Symbol] = []
    in_fence = False
    for index, line in enumerate(lines):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            symbols.append(
                Symbol(
                    name=truncate(title, 100),
                    kind=f"h{level}",
                    line=index + 1,
                    signature="#" * level + " " + truncate(title, 100),
                )
            )
    summary = ""
    for symbol in symbols:
        if symbol.kind == "h1":
            summary = symbol.name
            break
    return ParseResult(symbols=symbols, summary=summary)


def _parse_rst(text: str) -> ParseResult:
    lines = split_lines(text)
    symbols: list[Symbol] = []
    for index, line in enumerate(lines[:-1]):
        stripped = line.strip()
        underline = lines[index + 1].strip()
        if stripped and re.fullmatch(r"[=\-~^\"'`#*+]{3,}", underline) and len(underline) >= len(stripped) - 2:
            symbols.append(
                Symbol(name=truncate(stripped, 100), kind="section", line=index + 1, signature=stripped)
            )
    summary = symbols[0].name if symbols else ""
    return ParseResult(symbols=symbols, summary=summary)


def _parse_html(text: str) -> ParseResult:
    lines = split_lines(text)
    symbols: list[Symbol] = []
    for index, line in enumerate(lines):
        for match in re.finditer(r"<h([1-6])[^>]*>(.*?)</h\1>", line, re.IGNORECASE):
            title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            if title:
                symbols.append(
                    Symbol(name=truncate(title, 100), kind=f"h{match.group(1)}", line=index + 1, signature=title)
                )
    summary = symbols[0].name if symbols else ""
    return ParseResult(symbols=symbols, summary=summary)
