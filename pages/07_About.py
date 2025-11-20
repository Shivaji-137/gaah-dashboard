from __future__ import annotations

import streamlit as st
from components.layout import inject_css

def main() -> None:
    st.set_page_config(page_title="About the Model", layout="wide")
    inject_css()
    
    st.title("Current Research: Machine Learning Framework for Time-Resolved Mobility Edges")
    
    st.markdown(r"""

    This dashboard is part of my active independent research project titled **“Machine Learning Framework for Time-Resolved Mobility Edges in 1D Quasiperiodic Systems.”**

    ### Research Motivation
    The project investigates the dynamics of wavefunctions in quasiperiodic lattices to identify and classify mobility edges using machine learning. While static spectral properties (like IPR) are well-understood, the **time-resolved** behavior of wavepackets near mobility edges offers a rich, dynamical perspective on localization transitions. To learn more, see my <a href="https://github.com/Shivaji-137/gaah-dashboard/blob/master/gaahmodel_quasiperiodic_research.pdf" target="_blank">research plan (PDF)</a> (Machine learning methodology is left to write).

    ### Computational Approach
    - **System Sizes**: Simulations cover lattice lengths $L \in \{100, 200, 300, 400, 800, 1000, 1200, 1500, 1800, 2000, 4000, 5000, 10000\}$.
    - **Performance**: Efficient execution for smaller systems; parallel processing (hybrid-parallel pipeline) enables scaling up to $L=10000$, though computational cost and memory demands become significant for the largest lattice sizes.
    - **Data Generation**: The `gaahmodel_parallel_timeevol.py` pipeline generates comprehensive datasets containing eigenvalues, eigenvectors, and full time-evolution histories.

    ### Key Observables & Equations

    The analysis focuses on several critical metrics to characterize the phase of the system:

    #### 1. Wavefunction Probability Density
    The fundamental quantity tracking the particle's distribution across lattice sites $j$ at time $t$:
    $$ P(j, t) = |\psi_j(t)|^2 $$
    *   **Visualization**: Heatmaps of $|\psi_j(t)|^2$ reveal the spreading (ballistic/diffusive) or confinement (localization) of the wavepacket over time.

    #### 2. Mean Squared Displacement (MSD)
    Quantifies the spatial spread of the wavepacket from its center of mass:
    $$ \sigma^2(t) = \sum_j (j - \langle j \rangle)^2 |\psi_j(t)|^2 $$
    *   **Interpretation**:
        *   $\sigma^2(t) \sim t^2$: Ballistic transport (Extended phase).
        *   $\sigma^2(t) \sim t^\beta$ ($0 < \beta < 1$): Sub-diffusive (Critical/Multifractal).
        *   $\sigma^2(t) \sim \text{const}$: Saturation (Localized phase).

    #### 3. Survival Probability
    Measures the probability of the particle remaining at its initial position (typically the center $|L/2\rangle$):
    $$ P(t) = |\langle \psi(0) | \psi(t) \rangle|^2 $$
    *   **Interpretation**: A non-decaying finite value at long times indicates strong localization (memory of initial state).

    #### 4. Inverse Participation Ratio (IPR)
    A measure of the spatial extent of the wavefunction. For a normalized state:
    $$ \text{IPR}(t) = \sum_j |\psi_j(t)|^4 $$
    *   **Localized**: $\text{IPR} \sim O(1)$.
    *   **Extended**: $\text{IPR} \sim O(1/L)$.

    #### 5. Participation Entropy
    An information-theoretic measure of localization, often more sensitive to multifractal structures:
    $$ S_p(t) = -\sum_j |\psi_j(t)|^2 \ln |\psi_j(t)|^2 $$
    *   Provides an intuitive visualization of how information spreads through the lattice.

    ### Future Direction: Machine Learning Analysis
    The next phase involves processing these time-series datasets to train ML models. The goal is to:
    1.  **Classify Phases**: Distinguish between localized, extended, and critical regimes based purely on dynamical data.
    2.  **Detect Mobility Edges**: Predict the precise energy or parameter thresholds ($\lambda, \alpha$) where the transition occurs.
    3.  **Time-Resolved Inference**: Determine the minimum observation time required to accurately classify the system's phase.

    **Call for Collaboration**: I am actively seeking collaborators with expertise in Machine Learning and Deep Learning to help develop and refine these models. If you are interested in applying ML techniques to quantum many-body physics and dynamical phase transitions, please <a href="https://www.shivajichaulagain.com.np/#/contact" target="_blank">reach out to me</a> to collaborate on this exciting phase of the project.

    ## Physics Overview
    
    The **Generalized Aubry–André–Harper (GAAH)** model is a one-dimensional tight-binding model that exhibits a unique **mobility edge**—an energy threshold separating localized and extended eigenstates. Unlike the standard Anderson model (where all states are localized in 1D for any disorder) or the standard Aubry–André model (which has a global metal-insulator transition at a critical potential strength), the GAAH model allows for the coexistence of localized and extended states in the same spectrum, depending on the energy.

    This model was detailed by **Ganeshan, Pixley, and Das Sarma** in *Phys. Rev. Lett. 114, 146601 (2015)*.

    ## Hamiltonian and Equations

    The system is described by the following 1D tight-binding Hamiltonian:

    $$
    H = -t \sum_{n} (|n\rangle\langle n+1| + |n+1\rangle\langle n|) + \sum_{n} V_n |n\rangle\langle n|
    $$

    where:
    - **$n$** indexes the lattice sites.
    - **$t$** is the nearest-neighbor hopping amplitude (kinetic energy scale).
    - **$V_n$** is the quasiperiodic onsite potential.

    ### The Quasiperiodic Potential

    The specific onsite modulation $V_n$ is given by:

    $$
    V_n(\alpha, \lambda, \phi) = 2\lambda \frac{\cos(2\pi b n + \phi)}{1 - \alpha \cos(2\pi b n + \phi)}
    $$

    #### Term Definitions:
    - **$\lambda$ (Lambda)**: The strength of the quasiperiodic potential.
    - **$\alpha$ (Alpha)**: The deformation parameter, where $\alpha \in (-1, 1)$. 
        - When $\alpha = 0$, the model reduces to the standard Aubry–André (AA) model.
        - Non-zero $\alpha$ breaks the self-duality of the AA model in a controlled way, introducing an energy-dependent mobility edge.
    - **$b$**: An irrational number (frequency), typically the golden mean conjugate $b = (\sqrt{5}-1)/2 \approx 0.618$, ensuring the potential is incommensurate with the lattice.
    - **$\phi$ (Phi)**: A global phase offset, often used to generate ensemble statistics.

    ## The Exact Mobility Edge

    One of the most significant features of this model is the existence of an exact analytical expression for the mobility edge $E_c$. The boundary between localized and extended states is defined by the condition:

    $$
    \alpha E_c = 2 \text{sgn}(\lambda) (|t| - |\lambda|)
    $$

    - **Extended States**: Exist on one side of this energy boundary.
    - **Localized States**: Exist on the other side.
    
    This relationship allows for precise testing of numerical results (like IPR or Lyapunov exponents) against the theoretical prediction.

    ## Dashboard Guide: Physics & Analysis

    This dashboard provides a suite of tools to visualize and analyze the GAAH model's properties. Here is a guide to each tab and the underlying physics:

    ### 1. Spectral Analysis
    **What it does**: Visualizes the energy spectrum ($E_n$), level spacing statistics ($r$), and Inverse Participation Ratio (IPR) for individual eigenstates.
    
    **Physics**:
    - **Level Spacing Ratio ($r$)**: In the localized phase (Poisson statistics), energy levels are uncorrelated, and the mean ratio $\langle r \rangle \approx 0.386$. In the extended phase (GOE statistics), levels repel each other, leading to $\langle r \rangle \approx 0.530$. This metric is a robust indicator of the global phase.
    - **IPR Spectrum**: Plots IPR vs. Eigenstate Index. High IPR values ($\sim 1$) indicate localized states, while low values ($\sim 1/L$) indicate extended states.

    ### 2. Dynamics
    **What it does**: Tracks the time evolution of a wavepacket initially localized at the center of the lattice.
    
    **Physics**:
    - **Mean Squared Displacement (MSD)**: Shows how fast the particle spreads. Ballistic spreading ($\sigma^2 \propto t^2$) indicates extended states (metallic behavior). Saturation indicates localization (insulating behavior). Sub-diffusive growth suggests critical or multifractal states.
    - **Survival Probability**: A non-zero long-time limit means the particle never fully leaves its starting position, a hallmark of localization.

    ### 3. Mobility Edge
    **What it does**: Focuses on the energy-dependent transition between phases.
    
    **Physics**:
    - **IPR Scatter**: Plots IPR vs. Energy. You can visually identify the "mobility edge" where IPR values drop sharply from $\sim 1$ to $\sim 0$.
    - **Eigenstate Heatmap**: Visualizes $|\psi_n|^2$ for all states. Localized states appear as narrow bright spots; extended states look like waves spanning the system.
    - **Time-Evolution Heatmap**: Shows the probability density $|\psi(x,t)|^2$ in space-time. Vertical streaks indicate localization; spreading cones indicate transport.

    ### 4. Fractal Dimension
    **What it does**: Estimates the generalized fractal dimension $D_2$ from the scaling of IPR with system size $L$.
    
    **Physics**:
    - **Multifractality**: At the critical point (or mobility edge), wavefunctions are neither fully extended ($D_2=1$) nor fully localized ($D_2=0$). They are "multifractal," occupying a fractional dimension of the space. $D_2 \approx -\ln(\langle \text{IPR} \rangle)/\ln(L)$.

    ### 5. Global Trends
    **What it does**: Aggregates data across many simulations to show phase diagrams as a function of potential strength $\lambda$.
    
    **Physics**:
    - **Phase Diagram**: By plotting $\langle r \rangle$ or the fraction of localized states vs. $\lambda$, one can map out the entire phase diagram. The transition points match the theoretical predictions of the GAAH model.

    ## Numerical Implementation

    The results presented in this dashboard are generated using a custom hybrid-parallel pipeline (`gaahmodel_parallel_timeevol.py`). Key steps include:

    ### 1. Hamiltonian Construction
    - Built as a **sparse matrix** (CSR format) for memory efficiency.
    - Supports **Open** and **Periodic** boundary conditions.
    - The quasiperiodic potential is applied to the main diagonal, while hopping terms occupy the off-diagonals.

    ### 2. Diagonalization
    - Uses **`scipy.sparse.linalg.eigsh`** (Arnoldi/Lanczos) to efficiently find a subset of eigenvalues and eigenvectors for large systems.
    - Falls back to full diagonalization (`scipy.linalg.eigh`) for smaller system sizes or when the full spectrum is required.
    - **Observables Computed**:
        - **Inverse Participation Ratio (IPR)**: $\sum |\psi_n|^4$ to quantify localization.
        - **Level Spacing Ratio ($r$)**: To distinguish between Poisson (localized) and GOE (extended) level statistics.

    ### 3. Time Evolution (Dynamics)
    - **Method**: **Crank–Nicolson** scheme.
        - Equation: $(1 + i \frac{dt}{2} H) \psi(t+dt) = (1 - i \frac{dt}{2} H) \psi(t)$.
        - This implicit method is **unconditionally stable** and **unitary** (preserves the wavefunction norm).
    - **Initial Condition**: A particle localized at the center of the lattice ($|L/2\rangle$).
    - **Tracked Observables**:
        - **Mean Squared Displacement (MSD)**: $\sigma^2(t) = \sum (j - \langle j \rangle)^2 |\psi_j|^2$. Measures spreading speed.
        - **Survival Probability**: $P(t) = |\langle \psi(0) | \psi(t) \rangle|^2$. Measures return probability.

    ### 4. Parallelization
    - The pipeline uses `joblib` to parallelize tasks across multiple CPU cores, sweeping over parameters like $\lambda$, system size $L$, and phase $\phi$.

    > **Note**: If you identify any errors or potential issues in the Hamiltonian construction or the Crank-Nicolson time evolution code or physically inaccurate text or code implementation (e.g., results that seem physically inconsistent), please reach out to me. I am open to suggestions for modifications or updates to ensure the simulation's correctness.

    ## References

    1. **S. Ganeshan, J. H. Pixley, and S. Das Sarma**, "Nearest Neighbor Tight Binding Models with an Exact Mobility Edge in One Dimension", *Phys. Rev. Lett.* **114**, 146601 (2015).
    2. **S. Aubry and G. André**, *Ann. Israel Phys. Soc* **3**, 133 (1980).
    3. **Xiaopeng Li and S. Das Sarma**, "Mobility edges in one-dimensional bichromatic incommensurate potentials", *Phys. Rev. B* **96**, 085119 (2017).
    4. **Michele Modugno**, "Exponential localization in one-dimensional quasi-periodic optical lattices", *New J. Phys.* **11**, 033023 (2009).
    5. **David J. Luitz and Yevgeny Bar Lev**, "Many-body localization edge in the random-field Heisenberg chain", *Ann. Phys.* **529**, 1600350 (2017).
    6. **C. Dai, Y. Zhang, and J. Zhang**, "Floquet engineering of mobility edges in one-dimensional quasiperiodic lattices", *Phys. Rev. B* **103**, 014205 (2021).
    7. **Norman Y. Yao and Andrew C. Potter**, "Floquet localization and time crystals in driven quasiperiodic systems", *Annu. Rev. Condens. Matter Phys.* **10**, 233--252 (2019).
    8. **Juan Carrasquilla and Roger G. Melko**, "Machine learning phases of matter", *Nat. Phys.* **13**, 431--434 (2017).
    9. **Evert P. L. Van Nieuwenburg, Ye-Hua Liu, and Sebastian D. Huber**, "Learning phase transitions by confusion", *Nat. Phys.* **13**, 435--439 (2017).
    10. **Eliska Greplová and Evert P. L. van Nieuwenburg**, "Unsupervised identification of topological phase transitions using predictive models", *Nat. Mach. Intell.* **2**, 542--549 (2020).
    11. **Jordan Venderley, Vedika Khemani, and Eun-Ah Kim**, "Machine learning out-of-equilibrium phases of matter", *Phys. Rev. Lett.* **120**, 257204 (2018).
    12. **Frank Schindler, Nicolas Regnault, and Titus Neupert**, "Many-body localization transition with symmetry-protected topological states", *Phys. Rev. B* **95**, 245134 (2017).
    13. **Ren Yu, Xiaodong Gao, and Shun-Qing Shen**, "Identifying quantum phase transitions with unsupervised learning", *Phys. Rev. B* **102**, 085116 (2020).
    14. **DinhDuy Vu and Sankar Das Sarma**, "Generic mobility edges in several classes of duality-breaking one-dimensional quasiperiodic potentials", *Phys. Rev. B* **107**, 224206 (2023).
    15. **Madhumita Sarkar, Roopayan Ghosh, Arnab Sen, and K. Sengupta**, "Mobility edge and multifractality in a periodically driven Aubry-André model", *Phys. Rev. B* **103**, 184309 (2021).
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
