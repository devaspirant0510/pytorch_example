import torch

print("================= Reshape(View) ===================")
t = torch.FloatTensor([[[1,2,3],[4,5,6]],
                        [[7,8,9],[10,11,12]]])
print(f"tensor       : {t}")
print(f"tensor shape : {t.shape}")
# -1은 나머지 셰이프 값을 보고 알맞게 조정함
# ex 전체크기가 12(2x2x3)일경우 -1,4 를 줄경우 (3(-1),4) 로 만듦

vt = t.view(-1,4)
print(f"tensor       : {vt}")
print(f"tensor shape : {vt.shape}")

# 3차원으로도 reshape 가능 
# view( tensor 개수,height(행개수), width(열 개수))
print(f"tensor       : {vt.view(-1,6,1)}")
print(f"tensor shape : {vt.view(-1,6,1).shape}")
# 결국 12크기의 벡터를 view(reshape) 로 다양한 행렬을 만들수 있음 4x3x1,2x1x6,3x4 등등

t = torch.FloatTensor([[1,2,3],[4,5,6]])
print(t.squeeze())

print("==================== Squeeze ======================")
# 차원이 1인 차원을 모두 제거
rt = torch.rand(1,2,1,6)
print(f"tensor            : {rt}")
print(f"tensor shape      : {rt.shape}")
print(f"tensor -> squeeze : {rt.squeeze()}")
print(f"squeeze shape     : {rt.squeeze().shape}")
# dim=? 값을 주면 ? 에 shape가 1일경우 제거
# 1이 아니라면 그냥 무시함
print(f"tensor -> squeeze : {rt.squeeze(dim=2)}")
print(f"squeeze shape     : {rt.squeeze(dim=2).shape}")

print("=================== UnSqueeze =====================")
# Squeeze 와 반대로 지정한 dimension에 1을 추가
rt = torch.rand(3,4)
print(f"tensor          : {rt}")
print(f"tensor shape    : {rt.shape}")
print("dim = 0 에 unsquezee")
print(rt.unsqueeze(dim=0))
print(rt.unsqueeze(dim=0).shape,"\n")
print("dim = 1 에 unsquezee")
print(rt.unsqueeze(dim=1))
print(rt.unsqueeze(dim=1).shape,"\n")
print("dim = 2 에 unsquezee")
print(rt.unsqueeze(dim=2))
print(rt.unsqueeze(dim=2).shape,"\n")
# -1 을 줄경우 그냥 자동으로 가장 마지막에 1이 채워짐
print("dim = -1 에 unsquezee")
print(rt.unsqueeze(dim=-1))
print(rt.unsqueeze(dim=-1).shape,"\n")

print("================= Type Casting ===================")
print("LongTensor")
lt = torch.LongTensor([1,2,3,4])
print(f"tensor             : {lt}")
print(f"tensor type        : {lt.dtype}\n")
print("LongTensor => FloatTensor")
ft = lt.float() 
print(f"tensor             : {ft}")
print(f"tensor type        : {ft.dtype}\n")
print("ByteTensor")
bt = torch.ByteTensor([True,True,False,True])
print(f"tensor             : {bt}")
print(f"tensor type        : {bt.dtype}\n")
print("ByteTensor => IntTensor")
bt2it = bt.int()
print(f"tensor             : {bt2it}")
print(f"tensor type        : {bt2it.dtype}\n")
print("ByteTensor => DoubleTensor")
bt2lt = bt.double()
print(f"tensor             : {bt2lt}")
print(f"tensor type        : {bt2lt.dtype}\n")

print("================== Concatenate ====================")
x = torch.FloatTensor([[1,2],[3,4]])
y = torch.FloatTensor([[5,6],[7,8]])
#[행,열] 
# dim=0 행값을 기준으로 x와 y추가 즉 위아래로 추가
# dim=1 열값을 기준으로 x와 y추가 즉 양옆으로 추가
print(torch.cat([x,y],dim=0))
print(torch.cat([x,y],dim=1))

print("==================== Stacking ======================")
x = torch.FloatTensor([1,2])
y = torch.FloatTensor([3,4])
z = torch.FloatTensor([5,6])
# 기본값은 dim=0
print(torch.stack([x,y,z]))
print(torch.stack([x,y,z],dim=1))
# concate을 그대로 쓸경우 1차원 벡터이므로 2차원 행렬로 concatenate를 못함
print(torch.cat([x,y,z]))
# 해결방안
print(torch.cat([x.unsqueeze(0),y.unsqueeze(0),z.unsqueeze(0)]))
print(torch.cat([x.unsqueeze(1),y.unsqueeze(1),z.unsqueeze(1)],dim=1))

print("================== Ones and Zeros ====================")
x = torch.FloatTensor([[1,2,3],[4,5,6]])
print(f"Tensor             : {x}")
# zeros_like, ones_like 는 Torch.Tensor 를 입력받아 입력받은 Shpae 만큼 0 또는 1로 초기화 시킴
zl_x = torch.zeros_like(x)
print(f"Tensor zeros_like  : {zl_x}")
ol_x = torch.ones_like(x)
print(f"Tensor zeros_like  : {ol_x}")
# ones zeros 는 shape를 파라미터로 받아 0또는 1로채워진 행렬 생성
z_x = torch.zeros((4,4))
print(f"Tensor zeros((w,h)) : {z_x}")
o_x = torch.ones((4,4))
print(f"Tensor ones((w,h))  : {o_x}")

print("================ In-place Operation ==================")
x = torch.FloatTensor([[1,2],[3,4]])
out_x = x.mul(2)
print(out_x)
print(x)
# 연산결과를 저장할려면 변수에 메모리 할당을 해야되는데
# _ 을붙혀 inplace 옵션을 주면 파라미터로 받은값에 할당함
print(x.mul_(2))
print(x)