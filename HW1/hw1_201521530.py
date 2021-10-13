import numpy as np
import scipy.stats
# HW1 - 1
def discrete_asset_path(SO, mu, sigma, T, N):
    # configure dt
    dt = T/N
    Spath = np.zeros(shape=N+1)
    # configure initial value S0
    Spath[0] = S0
    for i in range(1, N+1):
        z = np.random.standard_normal()
        Spath[i] = Spath[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return Spath


#For BSM_prc_path, del_path

    # Call option pricing using BSM
def BSM_call_prc(S, K, tau, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * scipy.stats.norm.cdf(d1, 0, 1) - K * np.exp(-r * tau) * scipy.stats.norm.cdf(d2, 0, 1)
    # Changes in option value according to changes in underlying assets.
def BSM_call_delta(S, K, tau, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return scipy.stats.norm.cdf(d1, 0, 1)

# HW1 - 2
def BSM_prc_path(Spath, K, T, mu, r, sigma):
    N=Spath.size-1
    # configure dt
    dt = T/N
    Cpath=np.zeros(shape=N+1)
    # configure initial value C0 using BSM_call_prc
    Cpath[0] = BSM_call_prc(Spath[0], K, T, r, sigma)
    for i in range(1, N+1):
        z = np.random.standard_normal()
        Cpath[i]=BSM_call_prc(Spath[i], K, T - i * dt, r, sigma)
    return Cpath

# HW1 - 3
def del_path(Spath, K, T, mu, r, sigma, Pi):
    N=Spath.size-1
    dt = T/N
    port = np.zeros(shape=N+1)
    asset = np.zeros(shape=N+1)
    cash = np.zeros(shape=N+1)

    port[0]=Pi
    asset[0] = BSM_call_delta(Spath[0], K, T, r, sigma)
    # because of port = asset * S  + cash
    cash[0] = port[0] - (asset[0] * Spath[0])

    for i in range(1, N+1):
        z = np.random.standard_normal()
        port[i] = asset[i-1] * Spath[i] + (1 + r * dt) * cash[i-1]
        asset[i] = BSM_call_delta(Spath[i], K, T - i * dt, r, sigma)
        cash[i] = (1 + r * dt) * cash[i-1] + (asset[i-1] - asset[i]) * Spath[i]

    return port
# Configure Debug - pretend #ifdef
debug = 1

if debug == 0:
    import matplotlib.pyplot as plt
    
    S0 = 1
    mu = 0.05
    sigma = 0.2
    K = 1.2
    r = 0.05
    T = 5
    N = 1000

    spath = discrete_asset_path(S0,mu,sigma,T,N)
    cpath = BSM_prc_path(spath,K,T,mu,r,sigma)

    asset0 = BSM_call_delta(spath[0], K, T, r, sigma)
    cash0 = 1
    port0 = asset0 * spath[0] + cash0

    dpath = del_path(spath,K,T,mu,r,sigma, port0)
    
    plt.figure(figsize=(10, 10))
    plt.subplot(611)
    plt.plot(np.linspace(0, T, N+1), spath, color='b')
    plt.hlines(y=K, xmin=0, xmax=T, color='r', linestyles='dashed', label='Strike price')
    plt.ylabel('Asset path')

    plt.subplot(612)
    plt.plot(np.linspace(0, T, N+1), dpath, color='b', label='Portfolio value')
    plt.ylabel('Portfolio')
    plt.grid(True)
    plt.xlim((0, T))

    plt.subplot(613)
    plt.plot(np.linspace(0, T, N+1), cpath, color='b', label='Portfolio value')
    plt.ylabel('BSM_Call_Option')
    plt.grid(True)
    plt.xlim((0, T))

    plt.show()
else:
    quit