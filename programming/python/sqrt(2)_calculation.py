# This program demonstrates a way for calculating root 2 using continued fraction (https://en.wikipedia.org/wiki/Square_root_of_2).

from decimal import *
getcontext().prec=64;
n=1
d=1
n0=1
d0=1
r=Decimal(n/d)
i=0
while i<6:
	n0=n*n+2*d*d
	d0=2*n*d
	r=Decimal(n0/d0)
	print("A result of %d/%d is:"%(n0,d0))
	print(r)
	n=n0
	d=d0
	i+=1
