from flask import Flask, request, jsonify, render_template
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, partial_trace, entropy
import hashlib
import random

app = Flask(__name__, template_folder='templates')

# Positive ground (classical buffer)
positive_ground = []

# Quantum circuit for negative wires
negative_wires_circuit = None

def create_negative_wires():
    """Create a quantum circuit with negative phases (negative wires)."""
    qc = QuantumCircuit(3, 3)
    qc.h(1)
    qc.cx(1, 2)
    qc.z(1)  # Add negative phase to qubit 1
    return qc

def compute_quantum_metrics(state, qubit_indices=[1, 2]):
    """Compute entropy and eigenvalues for the entangled pair."""
    rho = partial_trace(state, [i for i in range(3) if i not in qubit_indices]).data
    ent = entropy(rho, base=2)
    eigenvalues = np.linalg.eigvals(rho).real
    return ent, eigenvalues

def encode_classical_to_quantum(bits):
    """Encode classical bits into a quantum state for feedback."""
    qc = QuantumCircuit(1, 1)
    if bits[0] == 1:
        qc.x(0)  # Prepare |1⟩
    if bits[1] == 1:
        qc.z(0)  # Add negative phase
    return qc

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    """Classical feedback to quantum circuit (positive to negative)."""
    global positive_ground, negative_wires_circuit
    bob_result = next((item for item in positive_ground if item['type'] == 'bob_result'), None)
    if not bob_result:
        return render_template('results.html', status='Error: No result to process', control_bits=None)

    # Use Bob’s result to control quantum circuit
    classical_bit = bob_result['classical_bit']
    bob_zero_prob = bob_result['bob_zero_prob']
    
    # Verify teleportation
    expected_prob = 1.0 if classical_bit == 1 else 0.853
    if abs(bob_zero_prob - expected_prob) > 0.1:
        return render_template('results.html', status='Error: Teleportation failed', control_bits=None)

    # Encode classical bits into quantum state
    c0, c1 = random.randint(0, 1), random.randint(0, 1)  # Simulate new control bits
    feedback_circuit = encode_classical_to_quantum([c0, c1])
    
    # Reinitialize negative wires with feedback
    negative_wires_circuit = create_negative_wires()
    negative_wires_circuit.compose(feedback_circuit, qubits=[0], inplace=True)
    
    # Store feedback in positive ground
    positive_ground.append({
        'type': 'feedback',
        'control_bits': [c0, c1]
    })
    
    return render_template('results.html', status='Feedback applied, circuit reinitialized', control_bits=[c0, c1])

if __name__ == '__main__':
    # Use Gunicorn or another production server for deployment
    app.run(debug=True)
