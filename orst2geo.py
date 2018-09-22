import itertools
import math
import sys

import orst

try:
    s = sys.argv[1]
except:
    s = sys.stdin.read()
    
ret = []
for elem in s:
    ret.append(orst.code_page.index(elem))

total = power = 0
for elem in ret:
    total += elem * 512 ** power
    power += 1

ret = []
while total:
    total, mod = divmod(total, 2320)
    ret.append(mod)
ret = ret[::-1]

grid = []
for i in range(4 * 29):
    grid.append([' '] * 20)

for index in ret:
    boxes, rem = divmod(index - 1, 4)
    line, box = divmod(boxes, 20)
    grid[line * 4 + rem][box] = 'O'

permutations = itertools.permutations(sorted(ret))
perm = 0
while True:
    p = next(permutations)
    if ret == list(p):
        break
    perm += 1

print(perm)
print(end = '\n'.join(map(''.join, grid)).rstrip())
