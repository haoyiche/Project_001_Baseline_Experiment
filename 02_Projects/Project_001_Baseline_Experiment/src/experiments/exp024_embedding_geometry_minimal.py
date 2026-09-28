import torch
import torch.nn.functional as F


A = torch.tensor([1.0, 0.0])
B = torch.tensor([10.0, 0.0])
C = torch.tensor([1.0, 1.0])


def l2(x, y):
    return torch.norm(x - y, p=2).item()


def cosine(x, y):
    return F.cosine_similarity(
        x.unsqueeze(0),
        y.unsqueeze(0),
    ).item()


print("=== RAW ===")

print("L2(A, B):", l2(A, B))
print("L2(A, C):", l2(A, C))

print("Cos(A, B):", cosine(A, B))
print("Cos(A, C):", cosine(A, C))


A_n = F.normalize(A, dim=0)
B_n = F.normalize(B, dim=0)
C_n = F.normalize(C, dim=0)


print()
print("=== NORMALIZED ===")

print("A_n:", A_n)
print("B_n:", B_n)
print("C_n:", C_n)

print("L2(A_n, B_n):", l2(A_n, B_n))
print("L2(A_n, C_n):", l2(A_n, C_n))

print("Cos(A_n, B_n):", cosine(A_n, B_n))
print("Cos(A_n, C_n):", cosine(A_n, C_n))