import numpy as np
from scipy.optimize import root_scalar, fsolve
import cvxpy as cp
import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(linewidth=200, precision=4, suppress=True)
Lambda = 0.5


def tanh(x):
    return np.tanh(Lambda * x)


def tanh_der(x):
    return Lambda * (1 - np.tanh(Lambda * x)**2)


def equilibrium(W1, V21, b1, W2, V12, b2, guess=None):
    dim1, dim2 = len(b1), len(b2)
    if guess == None:
        guess = np.zeros(dim1 + dim2)
    def equations(y):
        y1, y2 = y[:dim1], y[dim1:]
        err1 = y1 - tanh(W1 @ y1 + V21 @ y2 + b1)
        err2 = y2 - tanh(W2 @ y2 + V12 @ y1 + b2)
        return np.concatenate([err1, err2])
    sol = fsolve(equations, guess)
    return sol[:dim1], sol[dim1:]


def extended_system(W1, V21, b1, W2, V12, b2, z1, z2):
    n1, n2 = W1.shape[0], W2.shape[0]
    nx = 2*(n1+n2)
    n_xi = n1 + n2

    A = np.zeros((nx, nx))
    
    B = np.zeros((nx, n_xi))
    B[:2*n1, :n1] = np.vstack([np.eye(n1), np.eye(n1)])  
    B[2*n1:, n1:] = np.vstack([np.eye(n2), np.eye(n2)])

    Theta = np.zeros((n_xi, nx))
    Theta[:n1, n1:2*n1] = W1           
    Theta[:n1, 2*n1:2*n1+n2] = V21   
    Theta[n1:, :n1] = V12              
    Theta[n1:, 2*n1+n2:] = W2          

    b_ext = np.concatenate([b1, b2])
    z_ext = np.concatenate([z1, z1, z2, z2]) 
    
    c = Theta @ z_ext + b_ext
    
    return A, B, Theta, b_ext, c, n1, n2


def mu(c):

    def target(s):
        if abs(s) < 1e-8:
            return 0
        t = Lambda * (s + c)
        tc = Lambda * c
        return s * Lambda * (1 - np.tanh(t)**2) - (np.tanh(t) - np.tanh(tc))
    
    try:
        sol = root_scalar(target, bracket=[0.001, 5], method='bisect')
        s_max = sol.root
        phi_max = tanh(s_max + c) - tanh(c)
        return phi_max / s_max
    except:
        return tanh_der(c)


def nu(c, r):
    abs_c = abs(c)
    lambda_r = Lambda * r
    lambda_c = Lambda * abs_c
    
    if r <= abs_c + 1e-8:
        return Lambda * (1 - np.tanh(lambda_r)**2)
    
    num = np.tanh(lambda_r) - np.tanh(lambda_c)
    denom = r - abs_c
    return num / denom


def sector_bounds(c1, c2, Theta1, Theta2):
    n1, n2 = len(c1), len(c2)
    
    row_norm1 = np.sum(np.abs(Theta1), axis=1)
    M1_list, N1_list = [], []
    for i in range(n1):
        r = row_norm1[i] + abs(c1[i])
        M1_list.append(mu(c1[i]))
        N1_list.append(nu(c1[i], r))
    
    row_norm2 = np.sum(np.abs(Theta2), axis=1)
    M2_list, N2_list = [], []
    for i in range(n2):
        r = row_norm2[i] + abs(c2[i])
        M2_list.append(mu(c2[i]))
        N2_list.append(nu(c2[i], r))
    
    M1 = np.diag(M1_list)
    N1 = np.diag(N1_list)
    M2 = np.diag(M2_list)
    N2 = np.diag(N2_list)
    
    return M1, N1, M2, N2


def lmi(W1, V21, W2, V12, M1, N1, M2, N2):
    n1, n2 = M1.shape[0], M2.shape[0]
    
    H22 = cp.Variable((n1, n1), symmetric=True)
    H33 = cp.Variable((n2, n2), symmetric=True)
    H23 = cp.Variable((n1, n2))
    H11 = cp.Variable((n1, n1), symmetric=True)
    H44 = cp.Variable((n2, n2), symmetric=True)
    H41 = cp.Variable((n2, n1))
    
    Gamma1 = cp.Variable((n1, n1), diag=True)
    Gamma2 = cp.Variable((n2, n2), diag=True)
    
    L11 = -H22 - W1 @ M1 @ Gamma1 @ N1 @ W1
    L12 = -V21 @ N1 @ Gamma1 @ M1 @ W1 - H23.T
    L13 = Gamma1 @ (M1 + N1) @ W1 / 2
    L22 = H44 - H33 - V21 @ N1 @ Gamma1 @ M1 @ V21
    L23 = H41.T + Gamma1 @ (M1 + N1) @ V21 / 2
    L33 = H11 - Gamma1
    
    LMI1 = cp.bmat([
        [L11, L12.T, L13.T],
        [L12, L22, L23.T],
        [L13, L23, L33]
    ])
    
    M11 = H22 - H11 - V12 @ M2 @ Gamma2 @ N2 @ V12
    M12 = -W2 @ M2 @ Gamma2 @ N2 @ V12 - H41
    M13 = H23.T + Gamma2 @ (M2 + N2) @ V12 / 2
    M22 = -H44 - W2 @ M2 @ Gamma2 @ N2 @ W2
    M23 = Gamma2 @ (M2 + N2) @ W2 / 2
    M33 = H33 - Gamma2
    
    LMI2 = cp.bmat([
        [M11, M12.T, M13.T],
        [M12, M22, M23.T],
        [M13, M23, M33]
    ])
    
    constraints = [
        LMI1 << 0,
        LMI2 << 0,
        H22 >> 1e-6 * np.eye(n1),
        H33 >> 1e-6 * np.eye(n2),
        H11 >> 1e-6 * np.eye(n1),
        H44 >> 1e-6 * np.eye(n2),
        Gamma1 >> np.eye(n1),
        Gamma2 >> np.eye(n2),
    ]
    
    return constraints, Gamma1, Gamma2, H11, H22, H33, H44, H23, H41


def build_H(H11, H22, H33, H44, H23, H41, n1, n2):
    H_full = np.zeros((2*n1 + 2*n2, 2*n1 + 2*n2))

    H_full[:n1, :n1] = H11
    H_full[n1:2*n1, n1:2*n1] = H22
    H_full[2*n1:2*n1+n2, 2*n1:2*n1+n2] = H33
    H_full[2*n1+n2:, 2*n1+n2:] = H44
    H_full[n1:2*n1, 2*n1:2*n1+n2] = H23
    H_full[2*n1:2*n1+n2, n1:2*n1] = H23.T
    H_full[:n1, 2*n1+n2:] = H41.T
    H_full[2*n1+n2:, :n1] = H41
    
    return H_full


def build_Gamma(Gamma1_val, Gamma2_val):
    n1 = Gamma1_val.shape[0]
    n2 = Gamma2_val.shape[0]
    
    Gamma_full = np.zeros((n1 + n2, n1 + n2))
    Gamma_full[:n1, :n1] = Gamma1_val
    Gamma_full[n1:, n1:] = Gamma2_val
    
    return Gamma_full


def analyze_stability(W1, V21, b1, W2, V12, b2, z1=None, z2=None):
    print("="*60)
    print("АНАЛИЗ УСТОЙЧИВОСТИ ДВУХСЛОЙНОЙ RNN")
    print("="*60)

    if z1 is None or z2 is None:
        print("\n1. Поиск равновесия:")
        z1, z2 = equilibrium(W1, V21, b1, W2, V12, b2)
        print(f"z1 = {z1.round(4)}")
        print(f"z2 = {z2.round(4)}")
    else:
        print(f"\n1. Используем заданное равновесие:")
        print(f"z1 = {z1.round(4)}")
        print(f"z2 = {z2.round(4)}")

    c1 = W1 @ z1 + V21 @ z2 + b1
    c2 = W2 @ z2 + V12 @ z1 + b2

    print(f"\n2. Смещения (b1, b2):")
    print(f"b1 = {b1.round(4)}")
    print(f"b2 = {b2.round(4)}")

    print(f"\n2. Сдвиги (c1, c2):")
    print(f"c1 = {c1.round(4)}")
    print(f"c2 = {c2.round(4)}")

    print("\n3. Вычисление границ секторов:")
    Theta1 = np.hstack([W1, V21])
    Theta2 = np.hstack([W2, V12])
    
    M1, N1, M2, N2 = sector_bounds(c1, c2, Theta1, Theta2)
    print(f"M1 = \n{np.round(M1, 4)}")
    print(f"N1 = \n{np.round(N1, 4)}")
    print(f"M2 = \n{np.round(M2, 4)}")
    print(f"N2 = \n{np.round(N2, 4)}")

    print("\n4. Матрицы расширенной системы:")
    A, B, Theta, b_ext, c, n1, n2 = extended_system(W1, V21, b1, W2, V12, b2, z1, z2)
    print(f'\nA = \n{np.round(A, 4)}')
    print(f'\nB = \n{np.round(B, 4)}')
    print(f'\nTheta = \n{np.round(Theta, 4)}')

    print("\n5. Построение и решение LMI:")
    constraints, Gamma1, Gamma2, H11, H22, H33, H44, H23, H41 = lmi(W1, V21, W2, V12, M1, N1, M2, N2)
    
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)
    print("="*60)
    print(f"\nСтатус решения: {prob.status}")
    
    if prob.status == cp.OPTIMAL:
        print("Линейное матричное неравенство разрешимо.")
        print("Реккурентная нейронная сеть глобально экспоненциально устойчива.\n")
        
        Gamma1_val = Gamma1.value
        Gamma2_val = Gamma2.value
        if hasattr(Gamma1_val, 'diagonal'):
            Gamma1_val = np.diag(Gamma1_val.diagonal())
        elif hasattr(Gamma1_val, 'toarray'):
            Gamma1_val = Gamma1_val.toarray()
        if hasattr(Gamma2_val, 'diagonal'):
            Gamma2_val = np.diag(Gamma2_val.diagonal())
        elif hasattr(Gamma2_val, 'toarray'):
            Gamma2_val = Gamma2_val.toarray()
        if not isinstance(Gamma1_val, np.ndarray):
            Gamma1_val = np.array(Gamma1_val)
        if not isinstance(Gamma2_val, np.ndarray):
            Gamma2_val = np.array(Gamma2_val)
        n11 = Gamma1_val.shape[0]
        n22 = Gamma2_val.shape[0]
        Gamma = np.zeros((n11 + n22, n11 + n22))
        Gamma[:n11, :n11] = Gamma1_val
        Gamma[n11:, n11:] = Gamma2_val
        print(f"\nGamma =\n{np.round(Gamma, 4)}")

        H11_val = H11.value
        H22_val = H22.value
        H33_val = H33.value
        H44_val = H44.value
        H23_val = H23.value
        H41_val = H41.value
        n, N = M1.shape[0], M2.shape[0]
        H = build_H(H11_val, H22_val, H33_val, H44_val, H23_val, H41_val, n, N)
        print(f"\nH (функция Ляпунова) =\n{np.round(H, 4)}")
        print('\n'+"="*60)
        return True
    else:
        print("\n" + "="*60)
        print("Линейное матричное неравенство неразрешимо")
        print("Устойчивость не гарантируется")
        print("="*60)
        return False


if __name__ == "__main__":
    np.random.seed(42)
    
    W1 = np.array([[0.149, -0.0415], [0.1943, 0.4569]])
    V21 = np.array([[-0.0702, -0.0702], [0.4738, 0.2302]]) 
    b1 = np.array([-0.0756, 0.03531])
    
    W2 = np.array([[-0.139, -0.1397], [0.0726, -0.574]])
    V12 = np.array([[-3.8502, 0.0559], [1.1726, -0.5003]])  
    b2 = np.array([1.2668, -0.685])
    
    z1 = np.array([0.5282, 0.1791])
    z2 = np.array([-0.3366, -0.0697])
    
    stable = analyze_stability(W1, V21, b1, W2, V12, b2, z1, z2)