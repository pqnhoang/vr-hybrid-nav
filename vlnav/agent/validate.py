"""Program extraction, AST validation and safe loading (TASK_cap_mas.md §6).

A program is one `def task(api):` built from `api.<method>(...)` calls, literals and plain Python.
Not allowed: imports, `try`, `while`, `lambda`, `global`, attributes starting with `_`, names other
than `api`, local variables and SAFE_BUILTINS, nested functions or classes, generators. Every
`for` loops over a bounded iterable (a list, `range(<literal>)`, or an api query result).
"""
from __future__ import annotations

import ast
import builtins
import re

from vlnav.agent.api import ACTIONS, QUERIES

SAFE_BUILTINS = {n: getattr(builtins, n) for n in
                 ("len", "min", "max", "sorted", "abs", "range", "enumerate", "any", "all", "round",
                  "list", "set", "str", "int", "float", "bool", "zip", "reversed", "sum")}
FIELDS = {"name", "target", "room", "xy", "height", "score", "n_obs", "seen_s_ago", "verified",
          "label", "area", "objects", "yaw",
          # str / list methods programs commonly need
          "lower", "upper", "strip", "startswith", "endswith", "split", "join", "append", "extend",
          "index", "count", "replace"}
API_METHODS = set(QUERIES) | set(ACTIONS)
FORBIDDEN_NODES = (ast.Import, ast.ImportFrom, ast.Try, ast.While, ast.Lambda, ast.Global, ast.Nonlocal,
                   ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom, ast.With,
                   ast.AsyncFor, ast.AsyncWith, ast.Delete, ast.Raise)
try:
    FORBIDDEN_NODES += (ast.TryStar,)                         # Python >= 3.11
except AttributeError:
    pass


class ProgramError(ValueError):
    """The program does not parse or does not validate; the message goes back to the Coder."""


def extract_code(reply: str) -> str:
    """The fenced Python block of a Coder reply (the last one that defines task), else the reply."""
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply, re.DOTALL)
    for b in reversed(blocks):
        if "def task" in b:
            return b.strip() + "\n"
    if blocks:
        return blocks[-1].strip() + "\n"
    i = reply.find("def task")
    if i < 0:
        raise ProgramError("no `def task(api):` found in the reply")
    return reply[i:].strip() + "\n"


def validate(code: str) -> ast.Module:
    """Parse and check `code`; raises ProgramError with the line and the reason."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ProgramError(f"line {e.lineno}: syntax error: {e.msg}") from None
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    others = [n for n in tree.body if not isinstance(n, ast.FunctionDef)
              and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    if len(funcs) != 1 or funcs[0].name != "task" or others:
        raise ProgramError("the program must be exactly one function `def task(api):` and nothing else")
    fn = funcs[0]
    if [a.arg for a in fn.args.args] != ["api"] or fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs \
            or fn.decorator_list:
        raise ProgramError("the function must be `def task(api):`")
    assigned = {"api"}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            assigned.add(node.id)
        elif isinstance(node, ast.arg):
            assigned.add(node.arg)
    for node in ast.walk(fn):
        line = getattr(node, "lineno", "?")
        if isinstance(node, FORBIDDEN_NODES):
            raise ProgramError(f"line {line}: `{type(node).__name__}` is not allowed")
        if isinstance(node, ast.FunctionDef) and node is not fn:
            raise ProgramError(f"line {line}: nested functions are not allowed")
        if isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)):
            for g in node.generators:
                _bounded(g.iter, line)
        if isinstance(node, ast.For):
            _bounded(node.iter, line)
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ProgramError(f"line {line}: attribute `{node.attr}` is not allowed")
            if isinstance(node.value, ast.Name) and node.value.id == "api":
                if node.attr not in API_METHODS:
                    raise ProgramError(f"line {line}: `api.{node.attr}` does not exist; "
                                       f"use one of {sorted(API_METHODS)}")
            elif node.attr not in FIELDS:
                raise ProgramError(f"line {line}: attribute `{node.attr}` is not allowed")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in assigned and node.id not in SAFE_BUILTINS:
                raise ProgramError(f"line {line}: name `{node.id}` is not defined (only `api`, local "
                                   f"variables and {sorted(n for n in SAFE_BUILTINS if n[0].islower())})")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "api":
            _check_call(node, line)
    return tree


def _bounded(it: ast.expr, line) -> None:
    if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range":
        if not all(isinstance(a, ast.Constant) for a in it.args):
            raise ProgramError(f"line {line}: range() needs literal bounds")


def _check_call(node: ast.Call, line) -> None:
    name = node.func.attr
    if name in ("follow", "check", "fail") and not node.args and not node.keywords:
        raise ProgramError(f"line {line}: api.{name}() needs a text argument")
    if name in ("locate", "explore", "find") and node.args and isinstance(node.args[0], ast.Constant) \
            and not str(node.args[0].value).strip():
        raise ProgramError(f"line {line}: api.{name}() needs an object name")


def load_program(code: str):
    """Validated `task` function, executed in a fresh namespace with SAFE_BUILTINS only."""
    tree = validate(code)
    ns: dict = {"__builtins__": dict(SAFE_BUILTINS)}
    exec(compile(tree, "<program>", "exec"), ns)              # noqa: S102 - validated above
    return ns["task"]
