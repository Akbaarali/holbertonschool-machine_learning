#!/usr/bin/env python3
def cdf(self,k):
    if not isinstance(k,int):
        k=int(k)
    if k<0:
        return 0
    cem=0
    for i in range(0,k+1):
        cem+=self.pmf(i)
        
    return cem
