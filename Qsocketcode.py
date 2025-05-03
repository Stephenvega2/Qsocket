import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, partial_trace, entropy
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

# Quantum teleportation circuit
def create_teleportation_circuit():
    qc = QuantumCircuit(3, 3)  # 3 qubits: Alice's (q0), entangled pair (q1, q2), 3 classical bits
    
    # Step 1: Prepare a state to teleport (e.g., |ψ> = cos(θ/2)|0> + sin(θ/2)|1>, θ=π/4)
    theta = np.pi / 4
    qc.ry(theta, 0)  # Rotate Alice's qubit to a custom state
    
    # Step 2: Create entangled Bell pair between q1 (Alice) and q2 (Bob)
    qc.h(1)
    qc.cx(1, 2)
    
    # Step 3: Alice's operations for teleportation
    qc.cx(0, 1)  # CNOT with Alice's qubit as control
    qc.h(0)      # Hadamard on Alice's qubit
    
    # Step 4: Measure Alice's qubits (q0, q1) to get classical bits (verification key)
    qc.measure(0, 0)
    qc.measure(1, 1)
    
    # Step 5: Bob's corrections based on classical bits
    qc.cx(1, 2)  # If c1=1, apply X gate
    qc.cz(0, 2)  # If c0=1, apply Z gate
    
    # Step 6: Measure Bob's qubit (q2) to verify teleportation
    qc.measure(2, 2)
    
    return qc

# Compute entropy and eigenvalues for the entangled pair
def compute_quantum_metrics(state, qubit_indices=[1, 2]):
    # Get reduced density matrix for the entangled pair (q1, q2)
    rho = partial_trace(state, [i for i in range(3) if i not in qubit_indices]).data
    # Compute von Neumann entropy
    ent = entropy(rho, base=2)
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(rho).real
    return ent, eigenvalues

# Classical processing of quantum results (the "socket")
def classical_processing(counts, entropy, eigenvalues):
    print(f"Quantum Channel Metrics: Entropy={entropy:.3f}, Eigenvalues={eigenvalues}")
    
    # Process measurement counts
    total_shots = sum(counts.values())
    results = []
    for state, count in counts.items():
        # State format: "c2_c1_c0" (Bob's result, Alice's verification bits)
        bob_result = state[0]  # Bob's qubit measurement
        alice_bits = state[2:4]  # Alice's verification bits
        probability = count / total_shots
        print(f"State {state}: Bob's result={bob_result}, Alice's bits={alice_bits}, "
              f"Count={count}, Probability={probability:.3f}")
        results.append({
            'state': state,
            'bob_result': bob_result,
            'alice_bits': alice_bits,
            'probability': probability
        })
    
    # Verify teleportation success (Bob's results should match Alice's initial state)
    # Initial state: |ψ> = cos(π/8)|0> + sin(π/8)|1> ≈ 0.923|0> + 0.382|1>
    # Expected probabilities: P(0) ≈ 0.853, P(1) ≈ 0.147
    bob_zero_prob = sum(count / total_shots for state, count in counts.items() if state[0] == '0')
    print(f"Bob's P(|0>)={bob_zero_prob:.3f}, Expected ≈ 0.853")
    
    # Use results in classical workflow (e.g., store, analyze)
    classical_data = {
        'entropy': entropy,
        'eigenvalues': eigenvalues.tolist(),
        'results': results,
        'bob_zero_prob': bob_zero_prob
    }
    
    # Channel verification
    if entropy > 0.9:
        print("Quantum teleportation channel is reliable (high entropy)")
    else:
        print("Quantum channel may be degraded (low entropy)")
    
    return classical_data

def main():
    # Initialize teleportation circuit
    qc = create_teleportation_circuit()
    
    # Simulators
    statevector_sim = Aer.get_backend('statevector_simulator')
    qasm_sim = Aer.get_backend('qasm_simulator')
    
    # Run circuit to get statevector (for entropy/eigenvalues)
    result = execute(qc, statevector_sim).result()
    state = result.get_statevector()
    
    # Compute entropy and eigenvalues
    ent, eigvals = compute_quantum_metrics(state)
    
    # Run teleportation and measure
    result = execute(qc, qasm_sim, shots=1024).result()
    counts = result.get_counts()
    
    # Visualize results
    plot_histogram(counts)
    plt.show()
    
    # Use results as "socket" by passing to classical processing
    classical_data = classical_processing(counts, ent, eigvals)
    
    # Example: Use classical_data in further workflow
    print("\nClassical Data Output (usable in further processing):")
    print(f"Entropy: {classical_data['entropy']:.3f}")
    print(f"Bob's P(|0>): {classical_data['bob_zero_prob']:.3f}")
    print("Sample Results:", classical_data['results'][:2])  # Show first two for brevity

if __name__ == "__main__":
    main()
