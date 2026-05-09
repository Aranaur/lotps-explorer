# Check what the CCP formula currently produces
s = {'sens': 0.99, 'prev': 0.01, 'spec': 0.95, 'ppv': 0.1667}

brf = (
    f"$$\\begin{{aligned}}"
    f"P(D|+) &= \\frac{{P(+|D)P(D)}}{{P(+|D)P(D) + P(+|\\neg D)P(\\neg D)}} \\\\[1ex]"
    f"&= \\frac{{{s['sens']:.3f} \\times {s['prev']:.4f}}}"
    f"{{{s['sens']:.3f} \\times {s['prev']:.4f} + {(1-s['spec']):.3f} \\times {(1-s['prev']):.4f}}} \\\\[1ex]"
    f"&\\approx {s['ppv']*100:.1f}\\%"
    f"\\end{{aligned}}$$"
)
print("BRF output:")
print(repr(brf))
print()

# CCP formula - current state (with the extra backslashes from the file)
# Line 566 in the file: f"E(T) &= N \\sum_{{i=1}}^N \\frac{{1}}{{i}} = N \\cdot H_N \\\\\\"
# That's 6 actual backslash chars in source. In an f-string:
# \\ -> \ (times 3 pairs) -> \\\ in the string value
# But that can't be right because it would be an unterminated string.
# Let me just read the actual bytes from the file.

with open("paradox_server.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

line566 = lines[565]  # 0-indexed
line567 = lines[566]
line573 = lines[572]
print(f"Line 566: {repr(line566.strip())}")
print(f"Line 567: {repr(line567.strip())}")
print(f"Line 573: {repr(line573.strip())}")

# Now simulate what MathJax actually receives
# The f-string evaluates, then ui.HTML sends it to the browser
# MathJax needs: $$ \begin{aligned} E(T) &= ... \\ &\approx ... \end{aligned} $$
