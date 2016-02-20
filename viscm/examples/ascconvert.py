# Simple program to convert ASCII grid format test image to 2D matrix text file to be loaded by numpy

def concat(l):
    s=""
    for i in l:
        s=s+i+" "
    return s




print("Enter filename")
s = input()
t = open(s+".asc").read().splitlines()
ncols = int(t[0].split()[1])
nrows= int(t[1].split()[1])
t=t[6:]
n = open(s+".txt","w")
for r in range(nrows):
    print(concat(t[r*ncols:(r+1)*ncols]), file = n)