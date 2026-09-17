import numpy as np
import pandas as pd

Lambda = 0.5

def tanh(x):
    return np.tanh(Lambda * x)


def tanhder(x):
    return Lambda * (1 - np.tanh(Lambda * x)**2)


def mse(actual, predicted):
    return np.mean((actual - predicted)**2)


def msegrad(actual, predicted):
    return 2 * (predicted - actual) / actual.shape[0]


def init_params(input_dim = 3, hidden1 = 2, hidden2 = 2, output = 3, seed=42):
    np.random.seed(seed)
    return{
        'W1': np.random.randn(hidden1, hidden1) * 0.3,
        'V21': np.random.randn(hidden1, hidden2) * 0.3,
        'b1': np.random.randn(hidden1) * 0.1,
        'W2': np.random.randn(hidden2, hidden2) * 0.3,
        'V12': np.random.randn(hidden2, hidden1) * 0.3,
        'b2': np.random.randn(hidden2) * 0.1,
        'W_in1': np.random.randn(input_dim, hidden1) * 0.3,
        'W_in2': np.random.randn(input_dim, hidden2) * 0.3,
        'W_out': np.random.randn(hidden2, output) * 0.3,
        'b_out': np.random.randn(output) * 0.1
    }


def forward(x_seq, params):
    seq_len = x_seq.shape[0]
    h1_dim, h2_dim = params['W1'].shape[0], params['W2'].shape[0]
    h1 = np.zeros((seq_len, h1_dim))
    h2 = np.zeros((seq_len, h2_dim))
    outputs = np.zeros((seq_len, params['W_out'].shape[1]))
    for t in range(seq_len):
        h1_prev = h1[t-1] if t > 0 else np.zeros(h1_dim)
        h2_prev = h2[t-1] if t > 0 else np.zeros(h2_dim) 

        z1 = params['W1'] @ h1_prev + params['V21'] @ h2_prev + params['b1'] + x_seq[t] @ params['W_in1']
        h1[t] = tanh(z1)

        z2 = params['W2'] @ h2_prev + params['V12'] @ h1[t] + params['b2'] + x_seq[t] @ params['W_in2']
        h2[t] = tanh(z2)

        outputs[t] = h2[t] @ params['W_out'] + params['b_out']
    return h1, h2, outputs


def backwards(x_seq, y_seq, h1, h2, outputs, params, lr):
    seq_len = x_seq.shape[0]
    grads = {k: np.zeros_like(v) for k, v in params.items()}

    d_output = msegrad(y_seq, outputs)
    dh2_next = np.zeros(params['W2'].shape[0])
    dh1_next = np.zeros(params['W1'].shape[0])

    for t in range(seq_len - 1, -1, -1):
        grads['W_out'] += np.outer(h2[t], d_output[t])
        grads['b_out'] += d_output[t]

        dh2 = d_output[t] @ params['W_out'].T + dh2_next
        dz2 = dh2 * tanhder(params['W2'] @ (h2[t-1] if t > 0 else np.zeros_like(h2[t])) +
              params['V12'] @ h1[t] + params['b2'] + x_seq[t] @ params['W_in2'])

        grads['V12'] += np.outer(h1[t], dz2)
        grads['b2'] += dz2
        grads['W_in2'] += np.outer(x_seq[t], dz2)
        if t > 0:
            grads['W2'] += np.outer(h2[t-1], dz2)
            dh2_next = dz2 @ params['W2'].T
        else:
            dh2_next = np.zeros_like(dh2)
        
        dh1 = dz2 @ params['V12'].T + dh1_next
        dz1 = dh1 * tanhder(params['W1'] @ (h1[t-1] if t > 0 else np.zeros_like(h1[t])) + 
              params['V21'] @ (h2[t-1] if t > 0  else np.zeros_like(h2[t])) +
              params['b1'] + x_seq[t] @ params['W_in1'])
        
        grads['b1'] += dz1
        grads['W_in1'] += np.outer(x_seq[t], dz1)
        if t > 0:
            grads['W1'] += np.outer(h1[t-1], dz1)
            grads['V21'] += np.outer(h2[t-1], dz1)
            dh1_next = dz1 @ params['W1'].T
        else:
            dh1_next = np.zeros_like(dh1)

    for key in params:
        params[key] -= lr * grads[key]
    return params


def train(x_train, y_train, x_valid, y_valid, params, epochs=10000, lr=1e-3):
    for epoch in range(epochs):
        epoch_loss = 0
        for j in range(x_train.shape[0]):
            seq_x = x_train[j]
            seq_y = y_train[j]  
            seq_y_expanded = np.full((1, 1), seq_y[0]) 
            
            h1, h2, outputs = forward(seq_x, params)
            params = backwards(seq_x, seq_y_expanded, h1, h2, outputs, params, lr)
            epoch_loss += mse(seq_y_expanded, outputs)
        
        if epoch % 10 == 0:
            valid_loss = 0
            for j in range(x_valid.shape[0]):
                seq_x = x_valid[j]
                seq_y = y_valid[j]
                seq_y_expanded = np.full((7, 1), seq_y[0])
                _, _, outputs = forward(seq_x, params)
                valid_loss += mse(seq_y_expanded, outputs)
            #print(f"Epoch: {epoch} train loss {epoch_loss / len(x_train):.4f} valid loss {valid_loss / len(x_valid):.4f}")
    return params


def equilibrium(params, steps=500):
    h1 = np.zeros(params['W1'].shape[0])
    h2 = np.zeros(params['W2'].shape[0])
    for _ in range(steps):
        z1 = params['W1'] @ h1 + params['V21'] @ h2 + params['b1']
        h1 = tanh(z1)
        z2 = params['W2'] @ h2 + params['V12'] @ h1 + params['b2']
        h2 = tanh(z2)
    return h1, h2


def stability(params, z1, z2, n_tests=15, steps=1):
    print("\nПроверка устойчивости")
    h1 = z1 + np.random.randn(*z1.shape) * 2
    h2 = z2 + np.random.randn(*z2.shape) * 2
    for test in range(n_tests):
        for _ in range(steps):
            z1_curr = params['W1'] @ h1 + params['V21'] @ h2 + params['b1']
            h1 = tanh(z1_curr)
            z2_curr = params['W2'] @ h2 + params['V12'] @ h1 + params['b2']
            h2 = tanh(z2_curr)
        dist1 = np.linalg.norm(h1 - z1)
        dist2 = np.linalg.norm(h2 - z2)
        total_dist = np.linalg.norm(np.concatenate([h1 - z1, h2 - z2]))
        print(f"\nx1 = {np.round(h1, 4)}, расстояние до z1 = {dist1:.6f}")
        print(f"x2 = {np.round(h2, 4)}, расстояние до z2 = {dist2:.6f}")
    print()


if __name__ == "__main__":
    np.random.seed(42)
    
    dw = pd.read_csv('DailyDelhiClimateTest.csv')
    data = dw[['meantemp', 'humidity', 'meanpressure']].values
    
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data = (data - data_mean) / data_std

    seq_len = 1
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len]) 
        y.append(data[i+seq_len]) 
    X, y = np.array(X), np.array(y).reshape(-1, 1)

    split_train = int(0.9 * X.shape[0])
    split_valid = int(0.95 * X.shape[0])
    x_train, y_train = X[:split_train], y[:split_train]
    x_valid, y_valid = X[split_train:split_valid], y[split_train:split_valid]
    x_test, y_test = X[split_valid:], y[split_valid:]

    params = init_params()
    params = train(x_train, y_train, x_valid, y_valid, params)

    print("\nВеса после обучения:")
    print(f"W1 = {np.round(params['W1'], 4).tolist()}")
    print(f"V21 = {np.round(params['V21'], 4).tolist()}")
    print(f"b1 = {np.round(params['b1'], 4).tolist()}")
    print(f"W2 = {np.round(params['W2'], 4).tolist()}")
    print(f"V12 = {np.round(params['V12'], 4).tolist()}")
    print(f"b2 = {np.round(params['b2'], 4).tolist()}")
    
    z1, z2 = equilibrium(params)
    print(f"\nРавновесная точка:")
    print(f"z1 = {np.round(z1, 4)}")
    print(f"z2 = {np.round(z2, 4)}")

    stability(params, z1, z2)
    
    print()
    #print('-' * 100)
    print(f"{'вход':^20}|{'выход модели':^20}")
    #print(f"{'температура':^15}|{'влажность':^15}|{'давление':^15}||{'температура':^15}|{'влажность':^15}|{'давление':^15}")
    print('-' * 45)
    for j in range(x_train.shape[0]):
        seq_x = x_train[j]
        seq_y = y_train[j]  
            
        h1, h2, outputs = forward(seq_x, params)
        for t in range(len(seq_x)):
            real = seq_x[t][0] * data_std[0] + data_mean[0]
            out = outputs[t][0] * data_std[0] + data_mean[0]
            print(f"{real:^20.4f}|{out:^20.4f}")
            #print(f"{seq_x[t][0]:^15.4f}|{seq_x[t][1]:^15.4f}|{seq_x[t][2]:^15.4f}||{outputs[t][0]:^15.4f}|{outputs[t][1]:^15.4f}|{outputs[t][2]:^15.4f}")