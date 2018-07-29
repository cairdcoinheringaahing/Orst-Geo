import dictionary
import helpers
import orst

import os
import re
import sys
import time

switches = {

    lambda s: s[ 0] == ' ' and s != ' ': 0,
    lambda s: s[-1] == ' ' and s != ' ': 1,
    lambda s: s.istitle(): 2,
    lambda s: s.isupper(): 3,

}

mods = {

    lambda s: s.lower() in dictionary.short: 0,
    lambda s: s.lower() in dictionary.long : 1,
    lambda s: len(s) == 1 and 31 < ord(s) < 127: 2,
    lambda s: len(s) == 1 and s in orst.code_page and not(31 < ord(s) < 127): 3,

}

def compress_slow(string, start = 0):
    num = start
    while True:
        s = helpers.decompress(helpers.to_base(num, 510), orst.code_page)
        if s == string:
            break
        print('{:40} {}'.format(num, s))

        time.sleep(0.01)
        if num % 61 == 0:
            os.system('cls')
            print('Target:', string)
        
        num += 1
    return '"{}“'.format(''.join(orst.code_page[i] for i in helpers.to_base(num, 510)))

def compress_fast(tkns):
    tkns = tkns[::-1]
    total = 0

    def change(first_mul, first_add, sec_mul, sec_add):
        nonlocal total
        total *= first_mul
        total += first_add
        total *= sec_mul
        total += sec_add

    for tkn in tkns:
        if tkn.lower() in dictionary.short:
            switch = 4
            for func in switches:
                if func(tkn):
                    switch = switches[func]
                    break
            change(20453, dictionary.short.index(tkn.lower()), 1, 0)
            change(5, switch, 4, 0)

        elif tkn.lower() in dictionary.long:
            switch = 4
            for func in switches:
                if func(tkn):
                    switch = switches[func]
                    break
            change(227845, dictionary.long.index(tkn.lower()), 1, 0)
            change(5, switch, 4, 1)

        elif len(tkn) == 1 and 31 < ord(tkn) < 127:
            change(95, ord(tkn) - 32, 4, 2)

        else:
            change(512, orst.code_page.find(tkn), 4, 3)

    return '"{}“'.format(''.join(orst.code_page[i] for i in helpers.to_base(total, 510)))

print(compress_fast(['Hello', ',', ' ', 'World', '!']))
