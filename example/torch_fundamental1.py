import torch
# 1D Array pytorch
print("================= 1D Array Pytorch ===================")
tensor = torch.FloatTensor([1,2,3,4,5,6,7,8])
print(f"tensor  : {tensor}")
print(f"dim     : {tensor.dim()}")
print(f"shape   : {tensor.shape}")
print(f"indxing : {tensor[0],tensor[3]}")
print(f"slicing : {tensor[:3],tensor[1:3] }")

print("================= 2D Array Pytorch ===================")
tensor = torch.FloatTensor([[1,2,3],
                            [4,5,6],
                            [7,8,9]])
print(f"tensor  : {tensor}")
print(f"dim     : {tensor.dim()}")
print(f"shape   : {tensor.shape}")
print(f"indxing : {tensor[0,2],tensor[1,1]}")
print(f"slicing : {tensor[:,:2],tensor[1:3,1:3] }")

print("=================== Broadcasting =====================")
t1 = torch.FloatTensor([[1,5]])
t2 = torch.FloatTensor([[3,6]])
print(f"t1      : {t1}")
print(f"t2      : {t2}\n")
print(f"t1 + t2 : {t1+t2}")
print(f"t1 - t2 : {t1-t2}")
print(f"t1 x t2 : {t1*t2}")
print(f"t1 / t2 : {t1/t2}")

t3 = torch.FloatTensor([[3],[2]])
t4 = torch.FloatTensor([[1,2,3,6]])
print(f"t3      : {t3}")
print(f"t4      : {t4}\n")
"""
[[3,3,3,3],  [[1,2,3,6],
 [2,2,2,2]]   [1,2,3,6]]
"""
print(f"t3 + t4 : {t3+t4}")
print(f"t3 - t4 : {t3-t4}")
print(f"t3 x t4 : {t3*t4}")
print(f"t3 / t4 : {t3/t4}")

print("================ Mul VS Matrix Mul ===================")
print("---------------------Matrix Mul-----------------------")
# Matrix Mul 은 선형대수에서의 행렬 곱
m1 = torch.FloatTensor([[1,2],[4,6]])
m2 = torch.FloatTensor([[3],[7]])
print(f"m1        : {m1}")
print(f"m2        : {m2}")
print(f"m1 dot m2 : {m1.matmul(m2)}")
print(f"m1 shape : {m1.shape} m2 shape : {m2.shape} output shape : {(m1.matmul(m2)).shape}")
print("-------------------------Mul--------------------------")
# BroadCast 를 적용한 행렬 곱 shape를 맞춘후 곱함
print(f"m1        : {m1}")
print(f"m2        : {m2}")
print(f"m1 dot m2 : {m1.mul(m2)}")
# [[1,2],   [[3,3],
#  [4,6]]    [7,7]]
print(f"m1 shape : {m1.shape} m2 shape : {m2.shape} output shape : {(m1.mul(m2)).shape}")

print("======================= Mean ==========================")
t1 = torch.FloatTensor([[1,2,4]])
print(t1.mean())
try:
    t2 = torch.IntTensor([2,7])
    print(t2.mean())
except Exception as exc:
    print(exc)
    print("실수형 텐서만 평균을 구할수 있음")

t3 = torch.FloatTensor([[1,2,3],[4,5,6]])
print(f"t3            : {t3}")
print(f"t3 mean       : {t3.mean()}") # 전체에 대한 평균
print(f"t3 mean dim 0 : {t3.mean(dim=0)}") # axis 가 0 즉 열에대한 평균 
print(f"t3 mean dim 1 : {t3.mean(dim=1)}") # axis 가 1 즉 행에대한 평균 

print("======================= Sum ===========================")
t1 = torch.FloatTensor([[1,2,4]])
print(t1.sum())

t3 = torch.FloatTensor([[1,2,3],[4,5,6]])
print(f"t3            : {t3}")
print(f"t3 sum       : {t3.sum()}") # 전체에 대한 합
print(f"t3 sum dim 0 : {t3.sum(dim=0)}") # axis 가 0 즉 열에대한 합 
print(f"t3 sum dim 1 : {t3.sum(dim=1)}") # axis 가 1 즉 행에대한 합 

print("============== Max,ArgMax Min,ArgMin ==================")
t1 = torch.FloatTensor([[1,9,3],[4,5,6]])
# 기본적으로 max 함수를 쓸경우 전체 행렬에서 가장 큰 값을 찾음
print(f"t1       : {t1}")
print(f"t1 max   : {t1.max()}")

# max 함수에서 dim 값을 줄경우 리스트로 리턴하는데 첫번째는 max 값 두번째는 argMax 값을 리턴함
# max : 최댓값      argMax = 최대값이 있는 인덱스
print(f"t1 max dim 0    : {t1.max(dim=0)[0]}")
print(f"t1 argMax dim 0 : {t1.max(dim=0)[1]}")

print(f"t1 max dim 1    : {t1.max(dim=1)[0]}")
print(f"t1 argMax dim 1 : {t1.max(dim=1)[1]}")

print(f"t1       : {t1}")
print(f"t1 min   : {t1.min()}")

print(f"t1 min dim 0    : {t1.min(dim=0)[0]}")
print(f"t1 argMin dim 0 : {t1.min(dim=0)[1]}")

print(f"t1 min dim 1    : {t1.min(dim=1)[0]}")
print(f"t1 argMin dim 1 : {t1.min(dim=1)[1]}")
