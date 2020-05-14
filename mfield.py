import numpy as np 

def compute_direction(u, v, dudt, dvdt, f):
    x_0 = -dvdt.reshape(-1, 1)
    x_1 = dudt.reshape(-1, 1)
    x = np.concatenate((x_0, x_1), axis=1)
    y = (-dvdt * u + dudt * v).reshape(-1, 1)
    direction = np.dot(np.dot(np.linalg.inv(np.dot(x.T,  x)), x.T), y)
    return direction

def correct_data(u, v, dudt, dvdt, f, wx=0, wy=0, wz=0):
    dudt_corr = dudt - f * wy + wz * v + wx * u * v / f - wy * u * u / f
    dvdt_corr = dvdt + f * wx - wz * u - wy * u * v / f + wx * v * v / f
    return dudt_corr, dvdt_corr

def main():
    f = 10
    data1 = np.loadtxt('data/mfield1.txt', delimiter = ',')
    data2 = np.loadtxt('data/mfield2.txt', delimiter = ',')
    # problem 2 (a)
    u = data1[:, 0]
    v = data1[:, 1]
    dudt = data1[:, 2]
    dvdt = data1[:, 3]
    direction = compute_direction(u, v, dudt, dvdt, f)
    print('Focus of Expansion')
    print((direction / f).tolist())
    print('Direction of Translation')
    print(direction.tolist())
    # problem 2 (b)
    e1_ttc = (u - direction[0]) / dudt
    e2_ttc = (v - direction[1]) / dvdt
    ave_ttc = (e1_ttc + e2_ttc) / 2.0
    print('TIme to collision')
    print(ave_ttc)
    print('Average Time: ', np.mean(ave_ttc))
    # problem 2 (c)
    u = data2[:, 0]
    v = data2[:, 1]
    dudt = data2[:, 2]
    dvdt = data2[:, 3]
    dudt, dvdt = correct_data(u, v, dudt, dvdt, f, 0.1, 0.2, 0.3)
    direction = compute_direction(u, v, dudt, dvdt, f)
    print('Center of Motion')
    print((direction / f).tolist())
    print('Direction of Translation')
    print(direction.tolist())
    e1_ttc = (u - direction[0]) / dudt
    e2_ttc = (v - direction[1]) / dvdt
    ave_ttc = (e1_ttc + e2_ttc) / 2.0
    print('TIme to collision')
    print(ave_ttc)
    print('Average Time: ', np.mean(ave_ttc))


if __name__ == '__main__':
    main()

