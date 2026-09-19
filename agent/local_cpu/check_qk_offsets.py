import torch
def check(B,T,Hq,Hkv,D,C,prefill,pos0):
    packed=torch.arange(B*T*(Hq+2*Hkv)*D,dtype=torch.float64).reshape(B,T,-1)
    query=torch.full((B,Hq,T,D),-1.0,dtype=torch.float64); keys=torch.full((B,Hkv,C,D),-1.0,dtype=torch.float64); values=keys.clone()
    pf,qf,kf,vf=packed.reshape(-1),query.reshape(-1),keys.reshape(-1),values.reshape(-1)
    for row in range(B*T):
        batch,token=row//T,row%T
        for head in range(Hq+Hkv):
            packed_row=row*(Hq+2*Hkv)*D; offset=packed_row+head*D
            for col in range(D):
                x=pf[offset+col]
                if head<Hq: qf[((batch*Hq+head)*T+token)*D+col]=x
                else:
                    kv=head-Hq; pos=token if prefill else pos0
                    co=((batch*Hkv+kv)*C+pos)*D+col
                    kf[co]=x; vf[co]=pf[packed_row+(Hq+Hkv+kv)*D+col]
    q,k,v=packed.split((Hq*D,Hkv*D,Hkv*D),-1)
    q=q.reshape(B,T,Hq,D).transpose(1,2); k=k.reshape(B,T,Hkv,D).transpose(1,2); v=v.reshape(B,T,Hkv,D).transpose(1,2)
    sl=slice(0,T) if prefill else slice(pos0,pos0+1)
    assert torch.equal(query,q.contiguous()), "query layout"
    assert torch.equal(keys[:,:,sl],k) and torch.equal(values[:,:,sl],v), "kv layout"
    rest=torch.ones(C,dtype=torch.bool); rest[sl]=False
    assert (keys[:,:,rest]==-1).all() and (values[:,:,rest]==-1).all(), "tail touched"
for a in [(2,5,4,2,8,9,True,0),(3,7,8,2,16,11,True,0),(1,1,4,2,8,9,True,0),(2,1,4,2,8,9,False,6),(4,13,32,8,4,20,True,0)]: check(*a)
print("offset algebra OK for prefill (B>1, odd T) and decode")
