import numpy as np
# In instances_Coutsimilaire_ETA0_10.py

def calculate_F1(position):
    # Replace with the actual logic to calculate F1
    return sum(position[:len(position) // 2])

def calculate_F2(position):
    # Replace with the actual logic to calculate F2
    return sum(position[len(position) // 2:])


def instances_Coutsimilaire_ETA0_10(x):
    M = 1000
    B = 20
    n, m = 4, 7
    N = n + m
    p = np.array([56	,27,	52,	20,	28,	41,	24,	22	,29	,38,	27])
    a = np.array([0, 0, 0, 0, 0, 0, 0 ,0, 0 ,0,0])
    d = np.array([4000	,6000	,4000	,6000,	6000	,4000,	6000,	6000,	6000,	4000,	6000])
    f = np.array([120,	50	,120,	50	,50,	120,	50,	50	,50	,120,	50])
    h = np.array([377,	145,	383,	114,	168,	381	,140,	146,	151,	294,	179])
    w = np.array([
        [0,0,1,1, 0,0, 0, 0,0,0,0],
        [0,0,0,1, 0,1, 1, 1,0,0,0 ],
        [0,0,0,0, 1,0, 0, 0,0,0,0],
        [0,0,0,0, 0,0, 0, 1,0,0,0],
        [0,0,0,0, 0,0, 0, 0,0,0,0 ],
        [0,0,0,0, 0,0, 0, 1,0,0,0 ],
        [0, 0,0,0, 0,0, 0, 0,0,0,0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0,0,0, 0,0, 0, 0,0,0,0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0,0,0, 0,0, 0, 0,0,0,0],
    ])
    T, q = 60,  11
    bay=np.array([30,	36	,29,	48,	24,	33,	24,	28,	17,	48,	27])
    rh = [0.001, 0.0001, 0.001, 0.001, 0.001, 0.001, 0.0009, 0.001, 0.001, 0.001, 0.01, 0.001]
    ro = [0.00001, 0.00001, 0.000001, 0.00001, 0.0001, 0.0001]
    b, t, c = x[:N], x[N:2 * N], x[2 * N:3 * N]
    Xl, Yl, Dij, Iij = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))

    Cijk = np.zeros((n+m,max(bay), q))
#    matC = np.zeros((n + m, max(bay), q))  # Initialize matC here
#    matC = np.zeros((n + m, max(bay), q))

    #for e in range(n + m):
    #    matC[e] = np.zeros((bay[e], q))

    for i in range(N):
        for j in range(N):
            Xl[i, j] = 1 if t[j] >= t[i] + p[i] else 0
            Yl[i, j] = 1 if b[j] >= b[i] + h[i] and (
                        (t[i] <= c[j] and c[j] <= (t[i] + p[i])) or (t[i] <= t[j] and t[j] < (t[i] + p[i]))) else 0
            Dij[i, j] = 1 if w[i, j] == 1 and t[i] == t[j] else 0
            Iij[i, j] = 1 if w[i, j] == 1 and t[i] != t[j] else 0


    for e in range(n + m):
        for i in range(bay[e]):  # Iterate over the total number of bays
            column = np.random.randint(0, q)
            Cijk[e, i, column] = 1

#    for e in range(n + m):
#        matC[e] = x[3 * N + 4 * N ** 2: 3 * N + 4 * N ** 2 + bay[e] * q].reshape((bay[e], q))

    n_m = n + m
    matx = x[3 * n_m:3 * n_m + n_m ** 2].reshape(n_m, n_m)
    maty = x[3 * n_m + n_m ** 2:3 * n_m + 2 * n_m ** 2].reshape(n_m, n_m)
    matd = x[3 * n_m + 2 * n_m ** 2:3 * n_m + 3 * n_m ** 2].reshape(n_m, n_m)
    mati = x[3 * n_m + 3 * n_m ** 2:3 * n_m + 4 * n_m ** 2].reshape(n_m, n_m)

    #matC = x[3 * n_m + 4 * n_m ** 2:3 * n_m + 4 * n_m ** 2 + bay * q].reshape(bay, q)
    #for e in range(n+m):
    #    matC[e] = x[3 * n_m + 4 * n_m ** 2: 3 * n_m + 4 * n_m ** 2 + bay[e] * q].reshape((n+m,bay[e], q))

    #for e in range(n + m):
    #    matC[e] = x[3 * n_m + 4 * n_m ** 2: 3 * n_m + 4 * n_m ** 2 + bay[e] * q].reshape((bay[e], q))

#    for e in range(n + m):
#        matC[e] = np.zeros((bay[e], q))
#        matC[e] = x[3 * N + 4 * N ** 2: 3 * N + 4 * N ** 2 + bay[e] * q].reshape((bay[e], q))

    # inequality constraints
    C1 = np.zeros((n + m, n + m))
    C2 = np.zeros((n + m, n + m))
    C3 = np.zeros((n + m, n + m))
    C6 = np.zeros(n + m)
    C7 = np.zeros(n + m)
    C8 = np.zeros(n + m)
    C9 = np.zeros(n + m)
    C4 = np.zeros((n + m, n + m))
    C5 = np.zeros((n + m, n + m))
    C10 = np.zeros((n + m, n + m))
    # C11 = np.zeros((n+m, n+m)) # redundant
    C11a = np.zeros((n + m, n + m))
    C11b = np.zeros((n + m, n + m))
    # C12 = np.zeros((n+m, n+m)) # redundant
    C12a = np.zeros((n + m, n + m))
    C12b = np.zeros((n + m, n + m))
    C12c = np.zeros((n + m, n + m))

    for i in range(n + m):
        for j in range(n + m):
            if i < j:
                C1[i, j] = -Xl[i, j] - Xl[j, i] - Yl[i, j] - Yl[j, i] + 1
                C2[i, j] = Xl[i, j] + Xl[j, i] - 1
                C3[i, j] = Yl[i, j] + Yl[j, i] - 1
            if i != j:
                C4[i, j] = -t[j] + c[i] + (Xl[i, j] - 1) * M
                C5[i, j] = -b[j] + b[i] + h[i] + (Yl[i, j] - 1) * M
                C10[i, j] = Dij[i, j] + Iij[i, j] - 1
                if w[i, j] == 1:

                    # Updated C11 and C12, linearised them.
                    if t[i] - t[j] >= 0:
                        etaij = 1
                    else:
                        etaij = 0

                    C11a[i, j] = -(t[i] - t[j]) + Iij[i, j] * (2 * etaij - 1)

                    C11b[i, j] = -(t[j] - t[i]) + Iij[i, j] * (2 * (1 - etaij) - 1)

                    gamma = abs(t[i] - t[j])

                    if t[i] >= t[j]:
                        C12a[i, j] = (t[i] - t[j]) - gamma
                    # elif t[i] < t[j]:
                    else:
                        C12b[i, j] = (t[j] - t[i]) - gamma
                    C12c[i, j] = -(m * Iij[i, j]) + gamma

                    # old code
                    # C11[i, j] = -abs(t[i] - t[j]) + Iij[i, j]
                    # C12[i, j] = abs(t[i] - t[j]) - Iij[i, j] * M

    for i in range(n + m):
        C6[i] = -t[i] + a[i]
        C7[i] = -c[i] + t[i] + p[i]
    #   C8[i] = b[i] - B - h[i] + 1
        C9[i] = -b[i] + 1

    # constraints of QCs assignment
    CO2 = np.zeros((q - 1, q))
    #for e in range (n+m):
    CO1 = np.zeros((n+m, max(bay), q))
    CO3 = np.zeros((n+m, max(bay), q - 1))
    CO4 = np.zeros((n+m, max(bay) - 1, q - 1))
    CO5 = np.zeros((n+m, max(bay), q))

    for i in range(n + m):
        for j in range(bay[i]):
            jat1 = 0
            for k in range(q):
                jat1 += Cijk[i, j, k]
                CO1[i, j, k] = jat1 - 1

    #for k in range(2, q + 1):
    #    jat2 = 0
    #    for j in range(q - 1):
    #        jat2 += Cjk[j, k - 1]
    #        CO2[j, k - 1] = jat2

    #for k in range(1, q):
    #    jat3 = 0
    #    for j in range(bay + 1 - (q - k), bay + 1):
    #        jat3 += Cjk[j - 1, k - 1]
    #        CO3[j - 1, k - 1] = jat3

    for i in range(n + m):
        for j in range(2, bay[i] + 1):
            jat4 = 0
            for k in range(1, q):
                jat4 += Cijk[i, j - 1, k - 1]
            if j <= max(bay) - 1 and k <= q - 1:  # Vérifiez si les indices sont dans les limites
                CO4[i, j - 1, k - 1] = Cijk[i, j - 1, k - 1] - jat4

    for i in range(n + m):
        for j in range(2, bay[i] + 1):
            jat5 = 0
            for l in range(1, j):
                jat5 += Cijk[i, l - 1, k - 1]
            if j <= max(bay) and k <= q:  # Vérifiez si les indices sont dans les limites
                CO5[i, j - 1, k - 1] = Cijk[i, j - 1, k - 1] - jat5

        # Exterior Penalty Function Method calculations
    add1 = np.sum(C1[C1 > 0])
    add2 = np.sum(C2[C2 > 0])
    add3 = np.sum(C3[C3 > 0])
    add4 = np.sum(C4[C4 > 0])
    add5 = np.sum(C5[C5 > 0])
    add6 = np.sum(C6[C6 > 0])
    add7 = np.sum(C7[C7 > 0])
#    add8 = np.sum(C8[C8 > 0])
    add9 = np.sum(C9[C9 > 0])
    add10 = np.sum(np.abs(C10[C10 != 0]))
    # add11 = np.sum(C11[C11 > 0]) # redundant
    add11a = np.sum(C11a[C11a > 0])
    add11b = np.sum(C11b[C11b > 0])

    # add12 = np.sum(C12[C12 > 0]) # redundant
    add12a = np.sum(C12a[C12a > 0])
    add12b = np.sum(C12b[C12b > 0])
    add12c = np.sum(C12c[C12c > 0])

    # cranes calculations
    ad1 = np.sum(CO1[CO1 >= 1])
    ad2 = np.sum(CO2[CO2 >= 1])
    ad3 = np.sum(CO3[CO3 >= 1])
    ad4 = np.sum(CO4[CO4 >= 1])
    ad5 = np.sum(CO5[CO5 >= 1])

    # objective functions
    J = np.sum(c - a + f * np.maximum(0, (c - d)))

    JJ = np.sum(Cijk)

    # penalty factors
    rh1, rh2, rh3, rh4, rh5, rh6, rh7, rh8, rh9, rg10, rh11, rh12 = rh
    ro1, ro2, ro3, ro4, ro5, ro6 = ro

    # This is the old F1, before the linearisation is added.
    # The unconstrained function (objective function + penalty function)
    # F1 = J + (rh1 * np.maximum(0, add1)**2) + (rh2 * np.maximum(0, add2)**2) + (rh3 * np.maximum(0, add3)**2) + \
    #     (rh4 * np.maximum(0, add4)**2) + (rh5 * np.maximum(0, add5)**2) + (rh6 * np.maximum(0, add6)**2) + \
    #     (rh7 * np.maximum(0, add7)**2) + (rh8 * np.maximum(0, add8)**2) + (rh9 * np.maximum(0, add9)**2) + \
    #     (rg10 * np.abs(add10)**2) + (rh11 * np.maximum(0, add11)**2) + (rh12 * np.maximum(0, add12)**2)

    F1 = J + (rh1 * np.maximum(0, add1) ** 2) + (rh2 * np.maximum(0, add2) ** 2) + (rh3 * np.maximum(0, add3) ** 2) + \
         (rh4 * np.maximum(0, add4) ** 2) + (rh5 * np.maximum(0, add5) ** 2) + (rh6 * np.maximum(0, add6) ** 2) + \
         (rh7 * np.maximum(0, add7) ** 2)  + (rh9 * np.maximum(0, add9) ** 2) + \
         (rg10 * np.abs(add10) ** 2) + (rh11 * np.maximum(0, add11a) ** 2) + (rh11 * np.maximum(0, add11b) ** 2) + (
                     rh12 * np.maximum(0, add12a) ** 2) + (rh12 * np.maximum(0, add12b) ** 2) + (
                     rh12 * np.maximum(0, add12c) ** 2)

    F2 = JJ + (ro1 * np.abs(ad1) ** 2) + (ro2 * np.maximum(0, ad2) ** 2) + (ro3 * np.maximum(0, ad3) ** 2) + \
         (ro4 * np.maximum(0, ad4) ** 2) + (ro5 * np.maximum(0, ad5) ** 2) + (
                     ro6 * np.maximum(0, ad5) ** 2)  # Adjust if ro6 is used

    return np.array([F1, F2])


# for testing
if __name__ == "__main__":
    n, m = 5, 2
    T, bay, q = 25, 320, 4
    x = np.random.rand(3 * (n + m) + 4 * (n + m) * (n + m) + bay * q)
    x = np.random.rand(3 * (5 + 2) + 4 * (5 + 2) * (5 + 2) + 320 * 4)

    F = instances_Coutsimilaire_ETA0_10(x)
    print(F)