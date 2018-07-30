# Orst

The standard Hello, World! program is simply

    "Hello, World!"
    
However, we can shorten this by using Orst's builtin string compression with

    "BΡÁζŠ(ﬁ“
    
## Invocation

If the code is held in a file, `prog`, use the following invocation to fun the program

    $ python3 orst.py --file --unicode prog
    
With any additional inputs added to the end of the call. Otherwise, the code can be run be replacing
`--file` with `--cmd` and directly passing the code, rather than a file.
