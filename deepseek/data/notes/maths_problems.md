# Maths problems used (exact prompts; answers brute-forced)
Every prompt ends with the bench suffix:
```
Think it through carefully and check your work. End your response with a final line of the form:
ANSWER: <answer>
```

## tough (32,000-token budget)

### digit_sum_double — expected answer: `791`

Let S(n) denote the sum of the decimal digits of the positive integer n. How many integers n with 1 <= n <= 9999 satisfy S(n) = S(2n)?

### round_table_eyes — expected answer: `2618/59049`

Twelve people sit evenly spaced around a round table, so each person has a left neighbour, a right neighbour, and exactly one person directly across the table. Simultaneously, each person independently and uniformly at random chooses to look at their left neighbour, their right neighbour, or the person directly across. Two people make eye contact if they look at each other. What is the probability that nobody makes eye contact? Give the answer as a reduced fraction.

### coin_pattern_race — expected answer: `5/12`

A fair coin is tossed repeatedly until either the sequence HTTH or the sequence THTT appears as four consecutive tosses. What is the probability that HTTH appears first? Give the answer as a reduced fraction.

## held-out (16,000-token budget)

### code_trace — expected answer: `85`

What does this Python function return for n = 40?

```python
def f(n):
    a, b, out = 0, 1, 0
    for i in range(n):
        a, b = b, (a + b) % 97
        if (a * i) % 5 == 2:
            out += a
        elif b % 3 == 0:
            out -= i
    return out
```

### bitstrings — expected answer: `853`

How many binary strings of length 12 contain no three consecutive 1s and have an even number of 1s?

### dice_nine — expected answer: `21/2`

A fair six-sided die is rolled repeatedly until two consecutive rolls sum to 9. What is the expected number of rolls? Give the answer as a reduced fraction.

### numtheory — expected answer: `228`

How many integers n with 1 <= n <= 3000 satisfy both: n^2 + 1 is divisible by 13, and the sum of the decimal digits of n is odd?

### makespan — expected answer: `10`

Seven tasks have durations A=3, B=2, C=4, D=2, E=3, F=1, G=2 (hours). Precedence: C needs A; D needs A and B; E needs B; F needs C and D; G needs E and F. Two workers are available; each task is done by one worker without interruption, and a worker does one task at a time. What is the minimum total time (makespan) to finish all tasks?

### edit_distance — expected answer: `8`

What is the Levenshtein edit distance (insertions, deletions and substitutions each cost 1) between the strings "dynamicprogramming" and "dramaticprograms"?
