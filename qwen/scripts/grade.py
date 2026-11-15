# MATH-500 answer checking (Nov 15 2026): last \boxed{...} of the visible answer (after </think>; whole text if the
# thinking never closed), normalised string match, then a sympy equivalence check of the two LaTeX expressions.
import re
def last_boxed(s):
    i = s.rfind('\\boxed')
    if i < 0: return None
    j = s.find('{', i)
    if j < 0: return None
    d = 0
    for k in range(j, len(s)):
        if s[k] == '{': d += 1
        elif s[k] == '}':
            d -= 1
            if d == 0: return s[j + 1:k]
    return None
def norm(a):
    if a is None: return None
    a = a.strip().strip('$').strip()
    for x in ('\\left', '\\right', '\\!', '\\,', '\\;', '\\ ', '$', 'dfrac', 'tfrac', '^\\circ', '^{\\circ}', '\\%', '\\displaystyle'):
        a = a.replace(x, 'frac' if x in ('dfrac', 'tfrac') else '')
    a = re.sub(r'\\text\{\s*([^}]*)\}', r'\1', a); a = re.sub(r'\\mbox\{\s*([^}]*)\}', r'\1', a)
    a = re.sub(r'\s+', '', a).rstrip('.').replace('\\{', '{').replace('\\}', '}')
    a = re.sub(r'\\?frac\{([^{}]+)\}\{([^{}]+)\}', r'\1/\2', a)          # (\frac{a}{b} == a/b)
    a = re.sub(r'^(x|y|a|b|n|k|m)=', '', a) if '=' in a and a.count('=') == 1 and not ',' in a else a
    return a
def sym_eq(a, b):
    try:
        from sympy.parsing.latex import parse_latex
        import sympy
        x, y = parse_latex(a), parse_latex(b)
        return bool(sympy.simplify(x - y) == 0)
    except Exception:
        return False
def answer_of(text):
    vis = text.split('</think>')[-1] if '</think>' in text else text
    return last_boxed(vis)
def correct(text, expected):
    got = answer_of(text)
    if got is None: return False
    a, b = norm(got), norm(expected)
    if a == b: return True
    try:
        if abs(float(a) - float(b)) < 1e-6: return True
    except Exception: pass
    return sym_eq(a, b)
