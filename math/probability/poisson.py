#!/usr/bin/env python3
def pmf(self,k):
    if not isinstance(k, int):
        k=int(k)
        
    if k<0:
        return 0
    
    factorial=1
    for i in range(1,k+1):
        factorial*=i
    
    e_term=2.7182818285**(-self.lambtha)
    return (e_term*(self.lambtha**k))/factorial
