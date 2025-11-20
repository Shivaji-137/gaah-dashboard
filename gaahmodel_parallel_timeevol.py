#!/usr/bin/env python3
"""
Full pipeline for generalized Aubry–André–Harper (GAAH) model.

Features:
- Static spectral analysis: eigenvalues, eigenvectors, IPR, level-spacing ⟨r⟩
- Lyapunov exponent computation via transfer matrix
- Time evolution (Crank–Nicolson)
- Hybrid parallelization across (L, λ, bc, φ)
"""

import os
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
from joblib import Parallel, delayed
import multiprocessing

# -------------------------
# Model / Utility functions
# -------------------------

def build_gaah_hamiltonian(L, t1, lam, alpha, phi, bc='open', dtype=np.float64):
    j = np.arange(L)
    Vj = 2*lam * np.cos(2.0 * np.pi * alpha * j + phi) / (1.0 - alpha * np.cos(2.0 * np.pi * alpha * j + phi))
    H = sp.diags([Vj], [0], shape=(L, L), format='csr', dtype=dtype)
    H += sp.diags([-t1*np.ones(L-1), -t1*np.ones(L-1)], offsets=[-1,1], format='csr', dtype=dtype)
    if bc == 'periodic':
        H += sp.coo_matrix(([-t1, -t1], ([0, L-1], [L-1, 0])), shape=(L,L)).tocsr()
    return H

def compute_IPR_from_evecs(evecs):
    probs = np.abs(evecs)**2
    return np.sum(probs**2, axis=0)

def compute_r_from_eigenvals(E, tol=None):
    E = np.asarray(E)
    if E.size < 3: return np.nan, np.array([])
    s = np.diff(E)
    if tol is None: tol = np.finfo(float).eps * max(1.0, np.max(np.abs(E))) * 100
    r_list = [min(s[i], s[i-1])/max(s[i], s[i-1]) for i in range(1,len(s)) if s[i]>tol and s[i-1]>tol]
    return float(np.mean(r_list)) if r_list else np.nan, np.array(r_list)

def compute_lyapunov_exponent(E, t, V):
    """
    Lyapunov exponent for 1D tight-binding chain (transfer matrix method)
    E: energy
    t: hopping
    V: on-site potentials (1D array)
    """
    N = len(V)
    gamma = 0.0
    M = np.eye(2)
    for vn in V:
        T = np.array([[E - vn, -t],
                      [1.0,     0.0]])
        M = T @ M
        norm = np.linalg.norm(M)
        if norm>0:
            M /= norm
            gamma += np.log(norm)
    return gamma / (2.0*N)

# -------------------------
# Diagonalization helpers
# -------------------------

def diagonalize_full(H_sparse):
    H = H_sparse.toarray()
    E, V = la.eigh(H)
    return E, V

def diagonalize_sparse_center(H_sparse, k=400, sigma=0.0, tol=1e-8):
    L = H_sparse.shape[0]
    k_eff = min(k, max(2, L-2))
    if k_eff >= L-1: return diagonalize_full(H_sparse)
    try:
        E, V = spla.eigsh(H_sparse, k=k_eff, sigma=sigma, which='LM', tol=tol, return_eigenvectors=True)
        idx = np.argsort(E)
        return E[idx], V[:, idx]
    except Exception:
        return diagonalize_full(H_sparse)

# -------------------------
# Crank–Nicolson time evolution
# -------------------------

def crank_nicolson_time_evolution(H, psi0, dt=0.01, tmax=100.0, sample_steps=50):
    L = H.shape[0]
    psi = psi0.astype(np.complex128)
    times = np.arange(0, tmax, dt)

    A = sp.eye(L, dtype=np.complex128) + 0.5j*dt*H
    B = sp.eye(L, dtype=np.complex128) - 0.5j*dt*H
    solver = spla.factorized(A.tocsc())

    msd_list, surv_list, t_list, psi_samples, sample_times = [], [], [], [], []
    j = np.arange(L)

    for step, t in enumerate(times):
        psi = solver(B @ psi)
        psi /= np.linalg.norm(psi)

        if step % sample_steps == 0:
            prob = np.abs(psi)**2
            mean_j = np.sum(j*prob)
            msd_list.append(np.sum((j-mean_j)**2*prob))
            surv_list.append(np.abs(np.vdot(psi0, psi))**2)
            t_list.append(t)
            psi_samples.append(psi.copy())
            sample_times.append(t)

    return {'t': np.array(t_list), 'msd': np.array(msd_list), 'survival': np.array(surv_list),
            'psi_samples': np.array(psi_samples), 'psi_times': np.array(sample_times), 'dt': dt, 'tmax': tmax}

# -------------------------
# Worker task
# -------------------------

def worker_task(L, t1, lam, alpha, phi, bc, k_eigs_max, evolve=False):
    H = build_gaah_hamiltonian(L, t1, lam, alpha, phi, bc)
    E, V = diagonalize_sparse_center(H, k=min(k_eigs_max, max(2,L-2)))
    mean_r, _ = compute_r_from_eigenvals(E)
    IPRs = compute_IPR_from_evecs(V)

    # Lyapunov exponent computation
    Vpot = H.diagonal()
    lyap = np.array([compute_lyapunov_exponent(Ei, t1, Vpot) for Ei in E])

    dyn_results = None
    if evolve:
        psi0 = np.zeros(L, dtype=np.complex128)
        psi0[L//2] = 1.0
        dyn_results = crank_nicolson_time_evolution(H, psi0)

    return L, lam, bc, phi, mean_r, IPRs, E, V, lyap, dyn_results

# -------------------------
# Main pipeline
# -------------------------

def run_pipeline_gaah(output_dir="results_gaah", L_list=None, t1=1.0, lam_list=None,
                      num_phi=8, alpha=(np.sqrt(5)-1)/2, bc_arg='both',
                      k_eigs_max=400, n_jobs=None, evolve=False):

    os.makedirs(output_dir, exist_ok=True)
    if L_list is None: L_list = [100, 200]
    if lam_list is None: lam_list = np.linspace(0.5, 3.0, 10)
    bc_list = ['open','periodic'] if bc_arg=='both' else [bc_arg]
    n_jobs = multiprocessing.cpu_count() if n_jobs is None else n_jobs

    tasks = [(L, t1, lam, alpha, np.random.uniform(0,2*np.pi), bc, k_eigs_max, evolve)
             for L in L_list for lam in lam_list for bc in bc_list for _ in range(num_phi)]

    print(f"Running {len(tasks)} tasks with n_jobs={n_jobs}, evolve={evolve}")

    results = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(worker_task)(*task) for task in tasks
    )

    for L, lam, bc, phi, mean_r, IPRs, E, V, lyap, dyn in results:
        fname = f"L{L}_bc-{bc}_lam{lam:.3f}.npz"
        outpath = os.path.join(output_dir, fname)
        np.savez_compressed(outpath,
                            L=L, lam=lam, bc=bc, phi=phi,
                            mean_r=mean_r, IPRs=IPRs, E=E,
                            eigvecs=V, lyapunov=lyap,
                            dynamics=dyn if dyn is not None else {},
                            eig_method='eigsh')
        print(f"Saved {fname} | ⟨r⟩={mean_r:.4f}")

    print("✅ Pipeline completed.")

# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAAH spectral + dynamical pipeline")
    parser.add_argument("--out", type=str, default="results_gaah")
    parser.add_argument("--bc", type=str, default="both", choices=['open','periodic','both'])
    parser.add_argument("--t1", type=float, default=1.0)
    parser.add_argument("--lam_start", type=float, default=0.5)
    parser.add_argument("--lam_stop", type=float, default=10.0)
    parser.add_argument("--lam_step", type=float, default=1.0)
    parser.add_argument("--phi_samples", type=int, default=10)
    parser.add_argument("--k_eigs_max", type=int, default=400)
    parser.add_argument("--n_jobs", type=int, default=None)
    parser.add_argument("--evolve", action="store_true")
    args = parser.parse_args()

    lam_list = np.arange(args.lam_start, args.lam_stop+1e-12, args.lam_step)
    run_pipeline_gaah(output_dir=args.out,
                      L_list=[100,200,300],
                      t1=args.t1,
                      lam_list=lam_list,
                      num_phi=args.phi_samples,
                      alpha=(np.sqrt(5)-1)/2,
                      bc_arg=args.bc,
                      k_eigs_max=args.k_eigs_max,
                      n_jobs=args.n_jobs,
                      evolve=args.evolve)
