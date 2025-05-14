from flask import Flask, render_template, send_file
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, partial_trace, entropy
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
import io
import base64
from qiskit.visualization import circuit_drawer

app = Flask(__name__)

# Quantum teleportation circuit (same as provided)
def create_teleportation_circuit():
    qc = QuantumCircuit(3, 3)
    theta = np.pi / 4
    qc.ry(theta, 0)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.measure(1, 1)
    qc.cx(1, 2)
    qc.cz(0, 2)
    qc.measure(2, 2)
    return qc

# Compute quantum metrics
def compute_quantum_metrics(state, qubit_indices=[1, 2]):
    rho = partial_trace(state, [i for i in range(3) if i not in qubit_indices]).data
    ent = entropy(rho, base=2)
    eigenvalues = np.linalg.eigvals(rho).real
    return ent, eigenvalues

# Classical processing
def classical_processing(counts, entropy, eigenvalues):
    total_shots = sum(counts.values())
    results = []
    for state, count in counts.items():
        bob_result = state[0]
        alice_bits = state[2:4]
        probability = count / total_shots
        results.append({
            'state': state,
            'bob_result': bob_result,
            'alice_bits': alice_bits,
            'probability': probability
        })
    bob_zero_prob = sum(count / total_shots for state, count in counts.items() if state[0] == '0')
    channel_status = "Reliable (high entropy)" if entropy > 0.9 else "Degraded (low entropy)"
    return {
        'entropy': entropy,
        'eigenvalues': eigenvalues.tolist(),
        'results': results,
        'bob_zero_prob': bob_zero_prob,
        'channel_status': channel_status
    }

# Generate histogram and return as base64
def generate_histogram(counts):
    fig = plot_histogram(counts)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_str

# Generate circuit diagram as ASCII (for HTML display)
def generate_circuit_diagram(qc):
    return circuit_drawer(qc, output='text').single_string()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_simulation')
def run_simulation():
    # Initialize and run circuit
    qc = create_teleportation_circuit()
    statevector_sim = Aer.get_backend('statevector_simulator')
    qasm_sim = Aer.get_backend('qasm_simulator')
    
    # Get statevector for metrics
    result = execute(qc, statevector_sim).result()
    state = result.get_statevector()
    ent, eigvals = compute_quantum_metrics(state)
    
    # Run simulation for counts
    result = execute(qc, qasm_sim, shots=1024).result()
    counts = result.get_counts()
    
    # Process results
    classical_data = classical_processing(counts, ent, eigvals)
    
    # Generate visualizations
    histogram = generate_histogram(counts)
    circuit_diagram = generate_circuit_diagram(qc)
    
    return render_template('results.html', 
                         data=classical_data, 
                         histogram=histogram, 
                         circuit=circuit_diagram)

if __name__ == '__main__':
    app.run(debug=True)
