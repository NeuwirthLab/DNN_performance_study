import torch, time
dev='cuda'; bf16=torch.bfloat16
torch.backends.cudnn.benchmark=True
def bench(N,C,H,K, mf):
    x=torch.randn(N,C,H,H,device=dev,dtype=bf16).to(memory_format=mf)
    conv=torch.nn.Conv2d(C,K,3,padding=1,bias=False).to(dev,bf16).to(memory_format=mf)
    for _ in range(20): y=conv(x); y.sum().backward()
    torch.cuda.synchronize(); t=time.time()
    for _ in range(50):
        x.grad=None
        y=conv(x); y.sum().backward()
    torch.cuda.synchronize(); return (time.time()-t)/50*1000
shapes=[("L1 56^2 C64",256,64,56,64),("L2 28^2 C128",256,128,28,128),
        ("L3 14^2 C256",256,256,14,256),("L4 7^2 C512",256,512,7,512)]
print(f"{'stage':14s} {'NCHW ms':>9} {'NHWC ms':>9} {'speedup':>8}")
tn=to=0
for name,N,C,H,K in shapes:
    a=bench(N,C,H,K,torch.contiguous_format)
    b=bench(N,C,H,K,torch.channels_last)
    tn+=a; to+=b
    print(f"{name:14s} {a:9.3f} {b:9.3f} {a/b:7.2f}x")
print(f"{'TOTAL':14s} {tn:9.3f} {to:9.3f} {tn/to:7.2f}x")
