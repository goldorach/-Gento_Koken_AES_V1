from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import numpy as np

# Définir les constantes pour le texte chiffré et le texte en clair attendu
cipherText = [0x8B, 0x66, 0x68, 0xC2, 0x7D, 0x22, 0x61, 0x05, 0xA9, 0x17, 0xD6, 0x61, 0x41, 0xBC, 0x7B, 0x67]
expectedPlainText = [0xC4, 0x93, 0xE8, 0x4A, 0xAD, 0xD1, 0xC3, 0x03, 0x91, 0x3A, 0xBD, 0x57, 0xFE, 0x09, 0x79, 0x36]


start_key = [0x00] * 16  # Commencer avec des zéros pour le force brute
const_sBox = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]

const_invSBox = [
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d,
]

const_rcon = [
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
]

def int_to_bin(n):
    return format(n, '08b')

def initialize_qubits(circuit, qubits, value):
    for i, bit in enumerate(reversed(value)):
        if bit == '1':
            circuit.x(qubits[i])

def add_round_key(circuit, state_qubits, key_qubits):
    for i in range(len(state_qubits)):
        circuit.cx(key_qubits[i], state_qubits[i])

def rotate_bytes(circuit, qubits):
    for i in range(3):
        circuit.swap(qubits[i], qubits[i + 1])

def substitute_bytes(circuit, qubits, sbox):
    for i in range(4):
        byte_value = 0
        for j in range(8):
            qubit_index = circuit.find_bit(circuit.qubits[qubits[i*8 + j]]).index
            byte_value |= qubit_index << (7 - j)
        byte_value = sbox[byte_value]
        byte_bin = int_to_bin(byte_value)
        for j, bit in enumerate(reversed(byte_bin)):
            if bit == '1':
                circuit.x(qubits[i*8 + j])

def create_key_expansion_circuit(key):
    qc = QuantumCircuit(8 * 32)  # Utilisation de 32 qubits pour minimiser l'utilisation simultanee
    for i in range(16):
        initialize_qubits(qc, range(8 * i, 8 * (i + 1)), int_to_bin(key[i]))
    for i in range(16):
        qc.cx(i, 16 + i)
    for i in range(16, 176, 4):
        temp = list(range(i % 32, (i % 32) + 4))
        if i % 16 == 0:
            rotate_bytes(qc, temp)
            substitute_bytes(qc, temp, const_sBox)
            rcon_bin = int_to_bin(const_rcon[i // 16 - 1])
            initialize_qubits(qc, temp[:8], rcon_bin)
            qc.x(temp[0])
        for j in range(4):
            qc.cx((i - 16 + j) % 32, (i + j) % 32)
            qc.cx(temp[j], (i + j) % 32)
        if i >= 32:
            for j in range(4):
                qc.reset((i - 32 + j) % 32)
    qc.measure_all()
    return qc

def apply_sbox(circuit, qubits, sbox):
    for i in range(16):
        byte = qubits[i * 8:(i + 1) * 8]
        byte_value = 0
        for j in range(8):
            qubit_index = circuit.find_bit(circuit.qubits[byte[j]]).index
            byte_value |= qubit_index << (7 - j)
        sbox_value = sbox[byte_value]
        sbox_bin = int_to_bin(sbox_value)
        for j, bit in enumerate(reversed(sbox_bin)):
            if bit == '1':
                circuit.x(byte[j])

def create_subbytes_circuit(state):
    qc = QuantumCircuit(8 * 16)  # Utilisation de 16 octets de qubits au lieu de 128
    for i in range(16):
        initialize_qubits(qc, range(8 * i, 8 * (i + 1)), int_to_bin(state[i]))
        apply_sbox(qc, range(8 * i, 8 * (i + 1)), const_sBox)
    qc.measure_all()
    return qc

def create_invsubbytes_circuit(state):
    qc = QuantumCircuit(8 * 16)  # Utilisation de 16 octets de qubits au lieu de 128
    for i in range(16):
        initialize_qubits(qc, range(8 * i, 8 * (i + 1)), int_to_bin(state[i]))
        apply_sbox(qc, range(8 * i, 8 * (i + 1)), const_invSBox)
    qc.measure_all()
    return qc

def create_shiftrows_circuit(state):
    qc = QuantumCircuit(8 * 16)  # Utilisation de 16 octets de qubits au lieu de 128
    for i in range(16):
        initialize_qubits(qc, range(8 * i, 8 * (i + 1)), int_to_bin(state[i]))
    qc.swap(1 * 8, 5 * 8)
    qc.swap(5 * 8, 9 * 8)
    qc.swap(9 * 8, 13 * 8)
    qc.swap(2 * 8, 10 * 8)
    qc.swap(6 * 8, 14 * 8)
    qc.swap(3 * 8, 15 * 8)
    qc.swap(15 * 8, 11 * 8)
    qc.swap(11 * 8, 7 * 8)
    qc.measure_all()
    return qc

def create_invshiftrows_circuit(state):
    qc = QuantumCircuit(8 * 16)  # Utilisation de 16 octets de qubits au lieu de 128
    for i in range(16):
        initialize_qubits(qc, range(8 * i, 8 * (i + 1)), int_to_bin(state[i]))
    qc.swap(13 * 8, 9 * 8)
    qc.swap(9 * 8, 5 * 8)
    qc.swap(5 * 8, 1 * 8)
    qc.swap(2 * 8, 10 * 8)
    qc.swap(6 * 8, 14 * 8)
    qc.swap(3 * 8, 7 * 8)
    qc.swap(7 * 8, 11 * 8)
    qc.swap(11 * 8, 15 * 8)
    qc.measure_all()
    return qc

def create_gmul_circuit(a, b):
    qc = QuantumCircuit(8 * 4 + 1 + 8)  # Utilisation de 33 qubits réduite
    initialize_qubits(qc, range(8), int_to_bin(a))
    initialize_qubits(qc, range(8, 16), int_to_bin(b))
    result = list(range(16, 24))
    high_bit = 24
    modulo = list(range(25, 33))
    initialize_qubits(qc, modulo, int_to_bin(0x1B))
    for i in range(8):
        qc.cx(8 + i, result[i])  # Multiplication initiale
        qc.cx(0, high_bit)  # Calcul du high_bit
        qc.x(0)  # Préparation pour le shift
        for j in range(1, 8):
            qc.cx(j, j - 1)
        qc.cx(high_bit, 7)
        for j in range(8):
            qc.cx(high_bit, modulo[j])
            qc.cx(modulo[j], result[j])
        qc.x(8)
        for j in range(7, 0, -1):
            qc.cx(8 + j, 8 + j - 1)
        qc.reset(high_bit)
        for j in range(8):
            qc.reset(modulo[j])
    qc.measure_all()
    return qc

def precompute_gmul_table():
    simulator = Aer.get_backend('statevector_simulator')
    gmul_table = [[0 for _ in range(256)] for _ in range(256)]
    for a in range(256):
        for b in range(256):
            gmul_circuit = create_gmul_circuit(a, b)
            transpiled_circuit = transpile(gmul_circuit, simulator)
            qobj = assemble(transpiled_circuit)
            result = simulator.run(qobj).result()
            statevector = result.get_statevector()
            measured_value = np.argmax(np.abs(statevector))
            gmul_table[a][b] = measured_value
    return gmul_table

gmul_table = precompute_gmul_table()

def create_mixcolumns_circuit(state):
    qc = QuantumCircuit(8 * 16)
    for i in range(16):
        initialize_qubits(qc, range(8 * i, 8 * (i + 1)), int_to_bin(state[i]))
    for i in range(4):
        for j in range(4):
            byte = i * 4 + j
            byte_2 = gmul_table[0x02][state[byte]]
            byte_3 = gmul_table[0x03][state[(byte + 1) % 4]]
            byte_1 = state[(byte + 2) % 4]
            byte_0 = state[(byte + 3) % 4]
            qc.cx(byte_2, state[byte])
            qc.cx(byte_3, state[byte])
            qc.cx(byte_1, state[byte])
            qc.cx(byte_0, state[byte])
            qc.reset(byte_2)
            qc.reset(byte_3)
            qc.reset(byte_1)
            qc.reset(byte_0)
    qc.measure_all()
    return qc

def create_invmixcolumns_circuit(state):
    qc = QuantumCircuit(8 * 16)
    for i in range(16):
        initialize_qubits(qc, range(8 * i, 8 * (i + 1)), int_to_bin(state[i]))
    for i in range(4):
        for j in range(4):
            byte = i * 4 + j
            byte_e = gmul_table[0x0e][state[byte]]
            byte_b = gmul_table[0x0b][state[(byte + 1) % 4]]
            byte_d = gmul_table[0x0d][state[(byte + 2) % 4]]
            byte_9 = gmul_table[0x09][state[(byte + 3) % 4]]
            qc.cx(byte_e, state[byte])
            qc.cx(byte_b, state[byte])
            qc.cx(byte_d, state[byte])
            qc.cx(byte_9, state[byte])
            qc.reset(byte_e)
            qc.reset(byte_b)
            qc.reset(byte_d)
            qc.reset(byte_9)
    qc.measure_all()
    return qc

def create_increment_key_circuit(key, increment):
    qc = QuantumCircuit(8 * 16)
    for i in range(16):
        initialize_qubits(qc, range(8 * i, 8 * (i + 1)), int_to_bin(key[i]))
    carry = QuantumRegister(8)
    qc.add_register(carry)
    initialize_qubits(qc, carry, int_to_bin(increment))
    for i in range(15, -1, -1):
        for j in range(8):
            qc.cx(carry[j], 8 * i + j)
            qc.ccx(carry[j], 8 * i + j, carry[j])
    qc.measure_all()
    return qc

def create_decrypt_circuit(cipherText, key):
    expanded_keys = key_expansion(key)
    qc = QuantumCircuit(8 * 16 + 8 * 16)  # Utilisation de 128 qubits pour l'etat et 128 qubits pour la cle
    for i in range(16):
        initialize_qubits(qc, range(8 * i, 8 * (i + 1)), format(cipherText[i], '08b'))
    for i in range(16):
        initialize_qubits(qc, range(128 + 8 * i, 128 + 8 * (i + 1)), format(key[i], '08b'))
    add_round_key(qc, range(128), range(128, 256))
    for round in range(9, 0, -1):
        qc.compose(create_invshiftrows_circuit([qc.qubits[i] for i in range(128)]), inplace=True)
        qc.compose(create_invsubbytes_circuit([qc.qubits[i] for i in range(128)]), inplace=True)
        add_round_key(qc, range(128), range(128, 256))
        qc.compose(create_invmixcolumns_circuit([qc.qubits[i] for i in range(128)]), inplace=True)
        for i in range(16):
            qc.reset(128 + 8 * i)
            initialize_qubits(qc, range(128 + 8 * i, 128 + 8 * (i + 1)), format(expanded_keys[round * 16 + i], '08b'))
    qc.compose(create_invshiftrows_circuit([qc.qubits[i] for i in range(128)]), inplace=True)
    qc.compose(create_invsubbytes_circuit([qc.qubits[i] for i in range(128)]), inplace=True)
    add_round_key(qc, range(128), range(128, 256))
    qc.measure_all()
    return qc

def generate_sequential_aes_keys(start_key):
    key = start_key.copy()
    while True:
        yield key
        for i in range(15, -1, -1):
            key[i] = (key[i] + 1) % 256
            if key[i] != 0:
                break

def try_decrypt_aes(cipherText, expectedPlainText, start_key):
    backend = Aer.get_backend('statevector_simulator')
    key_generator = generate_sequential_aes_keys(start_key)
    for key in key_generator:
        decrypt_circuit = create_decrypt_circuit(cipherText, key)
        transpiled_circuit = transpile(decrypt_circuit, backend)
        qobj = assemble(transpiled_circuit)
        result = backend.run(qobj).result()
        statevector = result.get_statevector()
        decryptedText = np.round(np.real(statevector[:16])).astype(int)
        decryptedBytes = [int(decryptedText[i*8:(i+1)*8], 2) for i in range(16)]
        if decryptedBytes == expectedPlainText:
            print(f"Key found: {key}")
            return key, decryptedBytes
    print("No key found")
    return None, None

def test_quantum_functions():
    state = [0x32, 0x88, 0x31, 0xe0, 0x43, 0x5a, 0x31, 0x37, 0xf6, 0x30, 0x98, 0x07, 0xa8, 0x8d, 0xa2, 0x34]
    simulator = Aer.get_backend('statevector_simulator')
    
    # Creer et tester le circuit SubBytes
    print("Creation du circuit SubBytes...")
    subbytes_circuit = create_subbytes_circuit(state)
    transpiled_circuit = transpile(subbytes_circuit, simulator)
    qobj = assemble(transpiled_circuit)
    print("Execution du circuit SubBytes...")
    result = simulator.run(qobj).result()
    counts = result.get_counts()
    print("Resultat du circuit SubBytes:", counts)
    plot_histogram(counts)
    
    # Creer et tester le circuit InvSubBytes
    print("Creation du circuit InvSubBytes...")
    invsubbytes_circuit = create_invsubbytes_circuit(state)
    transpiled_circuit = transpile(invsubbytes_circuit, simulator)
    qobj = assemble(transpiled_circuit)
    print("Execution du circuit InvSubBytes...")
    result = simulator.run(qobj).result()
    counts = result.get_counts()
    print("Resultat du circuit InvSubBytes:", counts)
    plot_histogram(counts)
    
    # Creer et tester le circuit ShiftRows
    print("Creation du circuit ShiftRows...")
    shiftrows_circuit = create_shiftrows_circuit(state)
    transpiled_circuit = transpile(shiftrows_circuit, simulator)
    qobj = assemble(transpiled_circuit)
    print("Execution du circuit ShiftRows...")
    result = simulator.run(qobj).result()
    counts = result.get_counts()
    print("Resultat du circuit ShiftRows:", counts)
    plot_histogram(counts)
    
    # Creer et tester le circuit InvShiftRows
    print("Creation du circuit InvShiftRows...")
    invshiftrows_circuit = create_invshiftrows_circuit(state)
    transpiled_circuit = transpile(invshiftrows_circuit, simulator)
    qobj = assemble(transpiled_circuit)
    print("Execution du circuit InvShiftRows...")
    result = simulator.run(qobj).result()
    counts = result.get_counts()
    print("Resultat du circuit InvShiftRows:", counts)
    plot_histogram(counts)
    
    # Creer et tester le circuit MixColumns
    print("Creation du circuit MixColumns...")
    mixcolumns_circuit = create_mixcolumns_circuit(state)
    transpiled_circuit = transpile(mixcolumns_circuit, simulator)
    qobj = assemble(transpiled_circuit)
    print("Execution du circuit MixColumns...")
    result = simulator.run(qobj).result()
    counts = result.get_counts()
    print("Resultat du circuit MixColumns:", counts)
    plot_histogram(counts)
    
    # Creer et tester le circuit InvMixColumns
    print("Creation du circuit InvMixColumns...")
    invmixcolumns_circuit = create_invmixcolumns_circuit(state)
    transpiled_circuit = transpile(invmixcolumns_circuit, simulator)
    qobj = assemble(transpiled_circuit)
    print("Execution du circuit InvMixColumns...")
    result = simulator.run(qobj).result()
    counts = result.get_counts()
    print("Resultat du circuit InvMixColumns:", counts)
    plot_histogram(counts)

test_quantum_functions()
