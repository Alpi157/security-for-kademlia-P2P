from flask import Flask, render_template, jsonify
import threading
import time
import re
import random
import hashlib
import math
import pandas as pd
import numpy as np
import joblib

###############################################################################
# GLOBAL LOG STORAGE & HELPER
###############################################################################
dashboard_logs = []  # All logs
concurrent_lookup_logs = []  # Only [Concurrent Lookup] lines

def log(message):
    print(message)
    dashboard_logs.append(message)
    if "[Concurrent Lookup]" in message:
        concurrent_lookup_logs.append(message)

###############################################################################
# LOAD ML MODEL / SCALER
###############################################################################
MODEL_PATH = "random_forest_p2p.pkl"
SCALER_PATH = "scaler_p2p.pkl"

rf_model = None
scaler = None
try:
    rf_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    log("Trained model and scaler loaded successfully.")
except Exception as e:
    log("Error loading model/scaler: " + str(e))

###############################################################################
# SIMULATION CONSTANTS
###############################################################################
BITSPACE = 16
BUCKET_SIZE = 7
ALPHA = 3
MAX_ROUNDS = 20

DDOS_DURATION = 30
NUM_ATTACKERS = 10
NUM_VICTIMS = 200
CONGESTION_THRESHOLD = 10
THROTTLE_FACTOR = 0.2
NUM_NORMAL_REQUESTS = 10000
LATENCY_THRESHOLD = 0.5

SYBIL_COUNT = 200
POISONER_COUNT = 20
POISONING_DURATION = 30
ECLIPSE_COUNT = 112
ECLIPSE_DURATION = 30

LOOKUP_KEYS = ["fileA.txt", "fileB.mp4", "docC.pdf"]

###############################################################################
# GLOBAL NETWORK POINTER
###############################################################################
global_network = None

###############################################################################
# UTILS
###############################################################################
def generate_random_ip():
    return f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"

def xor_distance(a, b):
    return a ^ b

###############################################################################
# ML-BASED DDoS DETECTION
###############################################################################
def detect_ddos_ml(features):
    FEATURE_NAMES = ['attacker_id', 'victim_id', 'latency', 'attack_requests',
                     'failed_lookups', 'total_requests', 'peer_diversity', 'base_latency']
    vector = [features.get(k, 0) for k in FEATURE_NAMES]
    df = pd.DataFrame(np.array(vector).reshape(1, -1), columns=FEATURE_NAMES)
    if scaler is not None:
        df_scaled = scaler.transform(df)
    else:
        df_scaled = df.values
    if rf_model is not None:
        prediction = rf_model.predict(df_scaled)
        return int(prediction[0])
    else:
        # Fallback heuristic if model/scaler isn't loaded
        if features['latency'] > LATENCY_THRESHOLD or features['attack_requests'] > 5:
            return 1
        return 0

def print_detection(victim):
    log(f"DDoS Attack detected on Node {victim.node_id}")

###############################################################################
# NODE & NETWORK CLASSES
###############################################################################
class Node:
    def __init__(self, node_id, ip=None):
        self.node_id = node_id
        self.ip = ip if ip else generate_random_ip()
        self.data = {}
        self.buckets = {i: [] for i in range(BITSPACE)}
        self.attack_requests = 0
        self.failed_lookups = 0
        self.latency = 0.0
        self.request_queue = 0
        self.drop_rate = 0
        self.total_requests = 0
        self.base_latency = random.uniform(0.01, 0.05)
        self.is_sybil = False
        self.is_poisoner = False
        self.is_eclipse = False
        self.ddos_detected = False

    def bucket_index(self, target_id):
        d = xor_distance(self.node_id, target_id)
        if d == 0:
            return None
        return d.bit_length() - 1

    def update_routing_table(self, contact):
        if contact.node_id == self.node_id:
            return
        b_idx = self.bucket_index(contact.node_id)
        if b_idx is None:
            return
        bucket = self.buckets[b_idx]
        for i, existing in enumerate(bucket):
            if existing.node_id == contact.node_id:
                bucket.pop(i)
                bucket.append(contact)
                return
        if len(bucket) < BUCKET_SIZE:
            bucket.append(contact)
        else:
            if random.random() < 0.2:
                bucket.pop(0)
                bucket.append(contact)

    def get_all_contacts(self):
        all_contacts = []
        for b in self.buckets.values():
            all_contacts.extend(b)
        # Deduplicate
        return list({c.node_id: c for c in all_contacts}.values())

    def find_node(self, target_id):
        if self.is_poisoner:
            return []
        self.total_requests += 1
        self.request_queue += 1
        contacts = self.get_all_contacts()
        contacts.sort(key=lambda c: xor_distance(c.node_id, target_id))
        self.request_queue -= 1
        return contacts[:ALPHA]

    def find_value(self, key):
        return self.data.get(key, None)

    def store(self, key, value):
        self.data[key] = value

    def ping(self):
        if self.is_poisoner:
            return False
        return random.random() < 0.95

    def __repr__(self):
        label = ""
        if self.is_sybil:
            label = "[Sybil]"
        elif self.is_poisoner:
            label = "[Poisoner]"
        elif self.is_eclipse:
            label = "[Eclipse]"
        return f"Node({self.node_id}){label} IP:{self.ip}"

class Network:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node):
        if self.nodes:
            bootstrap = self.get_random_node()
            contacts = bootstrap.find_node(node.node_id)
            for c in contacts:
                node.update_routing_table(c)
                c.update_routing_table(node)
            for n in self.nodes.values():
                n.update_routing_table(node)
                node.update_routing_table(n)
        self.nodes[node.node_id] = node

    def remove_node(self, node_id):
        if node_id in self.nodes:
            del self.nodes[node_id]
            for n in self.nodes.values():
                for b in n.buckets.values():
                    b[:] = [c for c in b if c.node_id != node_id]

    def get_random_node(self):
        return random.choice(list(self.nodes.values()))

    def all_nodes(self):
        return list(self.nodes.values())

###############################################################################
# LOOKUP & STORE FUNCTIONS
###############################################################################
def iterative_lookup(network, start_node, target_id, alpha=ALPHA, max_rounds=MAX_ROUNDS):
    queried = set()
    shortlist = set(start_node.find_node(target_id))
    shortlist.add(start_node)
    best_distance = min((xor_distance(n.node_id, target_id) for n in shortlist), default=float('inf'))
    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        unqueried = sorted([n for n in shortlist if n.node_id not in queried],
                           key=lambda c: xor_distance(c.node_id, target_id))
        if not unqueried:
            break
        to_query = unqueried[:alpha]
        improvement = False
        for node in to_query:
            queried.add(node.node_id)
            new_contacts = node.find_node(target_id)
            for contact in new_contacts:
                if contact not in shortlist:
                    shortlist.add(contact)
                    dist = xor_distance(contact.node_id, target_id)
                    if dist < best_distance:
                        best_distance = dist
                        improvement = True
        if not improvement:
            break
    return shortlist, rounds

def store_value(network, key, value):
    key_hash = int(hashlib.sha1(key.encode()).hexdigest(), 16) % (2 ** BITSPACE)
    origin = network.get_random_node()
    shortlist, rounds = iterative_lookup(network, origin, key_hash)
    candidates = sorted(list(shortlist), key=lambda c: xor_distance(c.node_id, key_hash))
    store_nodes = candidates[:BUCKET_SIZE]
    for n in store_nodes:
        n.store(key, value)
    return store_nodes, rounds

def lookup_value(network, key, start_node, max_rounds=MAX_ROUNDS):
    key_hash = int(hashlib.sha1(key.encode()).hexdigest(), 16) % (2 ** BITSPACE)
    queried = set()
    shortlist = set(start_node.find_node(key_hash))
    shortlist.add(start_node)
    rounds = 0
    best_distance = min((xor_distance(n.node_id, key_hash) for n in shortlist), default=float('inf'))
    while rounds < max_rounds:
        rounds += 1
        unqueried = sorted([c for c in shortlist if c.node_id not in queried],
                           key=lambda c: xor_distance(c.node_id, key_hash))
        if not unqueried:
            break
        to_query = unqueried[:ALPHA]
        improvement = False
        for node in to_query:
            queried.add(node.node_id)
            val = node.find_value(key)
            if val is not None:
                return val, rounds
            new_contacts = node.find_node(key_hash)
            for contact in new_contacts:
                if contact not in shortlist:
                    shortlist.add(contact)
                    dist = xor_distance(contact.node_id, key_hash)
                    if dist < best_distance:
                        best_distance = dist
                        improvement = True
        if not improvement:
            break
    return None, rounds

###############################################################################
# CONCURRENT LOOKUPS
###############################################################################
def concurrent_lookups(network, keys, duration, interval=2):
    start_time = time.time()
    while time.time() - start_time < duration:
        for key in keys:
            start_node = network.get_random_node()
            val, rounds = lookup_value(network, key, start_node)
            if val is not None:
                log(f"[Concurrent Lookup] Found '{key}' from node {start_node.node_id} in {rounds} rounds.")
            else:
                log(f"[Concurrent Lookup] '{key}' not found from node {start_node.node_id}.")
        time.sleep(interval)

###############################################################################
# SIMULATION FUNCTIONS (No Mitigation) - AGGRESSIVE ATTACKS
###############################################################################
def simulate_normal_traffic(network, num_requests=NUM_NORMAL_REQUESTS):
    for _ in range(num_requests):
        key = f"normal_file_{random.randint(1, 100)}.txt"
        start_node = network.get_random_node()
        iterative_lookup(network, start_node,
                         int(hashlib.sha1(key.encode()).hexdigest(), 16) % (2 ** BITSPACE))

def simulate_ddos_attack(network, duration=DDOS_DURATION, num_attackers=NUM_ATTACKERS, num_victims=NUM_VICTIMS,
                         keys=LOOKUP_KEYS):
    all_nodes = network.all_nodes()
    if len(all_nodes) < (num_attackers + num_victims):
        log("Not enough nodes for DDoS attack.")
        return
    attackers = random.sample(all_nodes, num_attackers)
    remaining = [n for n in all_nodes if n not in attackers]
    victims = random.sample(remaining, num_victims)
    log(f"\nInitiating DDoS Attack for {duration}s")
    log(f"   Attackers: {[a.node_id for a in attackers]}")
    log(f"   Victims:   {[v.node_id for v in victims]}")
    t = threading.Thread(target=concurrent_lookups, args=(network, keys, duration))
    t.start()
    t0 = time.time()
    # Increase iterations for more aggressive attack (from 5 to 10)
    while time.time() - t0 < duration:
        for _ in range(10):
            attacker = random.choice(attackers)
            victim = random.choice(victims)
            random_key = "target_file.txt"
            t_id = int(hashlib.sha1(random_key.encode()).hexdigest(), 16) % (2 ** BITSPACE)
            s0 = time.time()
            iterative_lookup(network, attacker, t_id)
            s1 = time.time()
            lat = (s1 - s0) + random.uniform(0.05, 0.3)
            victim.attack_requests += 1
            victim.latency += lat
            feats = {
                'attacker_id': attacker.node_id,
                'victim_id': victim.node_id,
                'latency': lat,
                'attack_requests': victim.attack_requests,
                'failed_lookups': victim.failed_lookups,
                'total_requests': victim.total_requests,
                'peer_diversity': len(victim.get_all_contacts()),
                'base_latency': victim.base_latency
            }
            pred = detect_ddos_ml(feats)
            if pred == 1 and not victim.ddos_detected:
                print_detection(victim)
                victim.ddos_detected = True
        time.sleep(0.1)
    t.join()
    log("\nDDoS Attack Summary:")
    for v in victims:
        log(f"Victim Node {v.node_id} -> Attack Requests: {v.attack_requests}, Total Latency: {v.latency:.4f}s")
    for v in victims:
        v.ddos_detected = False

def simulate_sybil_attack(network, num_sybil=100):
    # Increase number of sybil identities to 100 (more aggressive)
    attacker = network.get_random_node()
    log(f"\nInitiating Sybil Attack: Attacker Node {attacker.node_id} creates {num_sybil} sybil identities.")
    sybil_nodes = []
    for _ in range(num_sybil):
        nid = random.randint(0, 2 ** BITSPACE - 1)
        while nid in network.nodes:
            nid = random.randint(0, 2 ** BITSPACE - 1)
        ip = f"10.0.0.{random.randint(1, 254)}"
        n = Node(nid, ip=ip)
        n.is_sybil = True
        network.add_node(n)
        sybil_nodes.append(n)
    total = 0
    sybil_total = 0
    for node in network.all_nodes():
        for b in node.buckets.values():
            for c in b:
                total += 1
                if c.is_sybil:
                    sybil_total += 1
    perc = (sybil_total / total * 100) if total else 0
    log(f"\nSybil Attack Summary: {len(sybil_nodes)} sybil nodes added.")
    log(f"Routing Table Infiltration: {sybil_total}/{total} entries ({perc:.2f}%) are sybil nodes.")

def simulate_routing_table_poisoning(network, num_poisoners=20, duration=POISONING_DURATION,
                                     keys=LOOKUP_KEYS):
    # Increase number of poisoners to 20 (more aggressive)
    log(f"\nInitiating Routing Table Poisoning attack for {duration}s by {num_poisoners} poisoners.")
    for node in network.all_nodes():
        for b_idx in range(BITSPACE):
            bucket = node.buckets[b_idx]
            num_replace = len(bucket) // 2
            for i in range(num_replace):
                fake_id = random.randint(0, 2 ** BITSPACE - 1)
                while fake_id in network.nodes:
                    fake_id = random.randint(0, 2 ** BITSPACE - 1)
                fc = Node(fake_id)
                fc.is_poisoner = True
                bucket[i] = fc
    t = threading.Thread(target=concurrent_lookups, args=(network, keys, duration))
    t.start()
    time.sleep(duration)
    t.join()
    total = 0
    pz = 0
    for node in network.all_nodes():
        for b in node.buckets.values():
            for c in b:
                total += 1
                if c.is_poisoner:
                    pz += 1
    perc = (pz / total * 100) if total else 0
    log(f"\nRouting Table Poisoning Summary: {pz} out of {total} entries ({perc:.2f}%) are poisoned.")

def simulate_eclipse_attack(network, duration=ECLIPSE_DURATION, num_eclipse_nodes=50, keys=LOOKUP_KEYS):
    # Increase number of eclipse nodes to 50 (more aggressive)
    victim = network.get_random_node()
    log(f"\nInitiating Eclipse Attack on Node {victim.node_id} for {duration}s using {num_eclipse_nodes} eclipse nodes.")
    e_nodes = []
    for _ in range(num_eclipse_nodes):
        nid = random.randint(0, 2 ** BITSPACE - 1)
        while nid in network.nodes:
            nid = random.randint(0, 2 ** BITSPACE - 1)
        en = Node(nid)
        en.is_eclipse = True
        e_nodes.append(en)
    for b_idx in range(BITSPACE):
        victim.buckets[b_idx] = [random.choice(e_nodes) for _ in range(BUCKET_SIZE)]
    t = threading.Thread(target=concurrent_lookups, args=(network, keys, duration))
    t.start()
    time.sleep(duration)
    t.join()
    total = 0
    ec = 0
    for b in victim.buckets.values():
        for c in b:
            total += 1
            if c.is_eclipse:
                ec += 1
    perc = (ec / total * 100) if total else 0
    log(f"\nEclipse Attack Summary on Node {victim.node_id}: {ec}/{total} entries ({perc:.2f}%) are eclipse nodes.")

###############################################################################
# SUMMARY FUNCTION: PARSE LOGS FOR KEY METRICS
###############################################################################
def compute_summary():
    """
    Parse dashboard_logs to extract relevant metrics for the no-mitigation dashboard.
    The table expects:
      - network_size
      - total_lookups
      - lookup_success_rate
      - lookup_failure_rate
      - ddos_detections
      - sybil_nodes_added
      - concurrent_found
      - concurrent_not_found
      - ddos_victims_count
      - ddos_total_latency
      - ddos_avg_latency
    """
    summary = {
        "network_size": len(global_network.nodes) if global_network else 0,
        "total_lookups": 0,
        "lookup_success_rate": 0.0,
        "lookup_failure_rate": 0.0,
        "ddos_detections": 0,
        "sybil_nodes_added": 0,
        "concurrent_found": 0,
        "concurrent_not_found": 0,
        "ddos_victims_count": 0,
        "ddos_total_latency": 0.0,
        "ddos_avg_latency": 0.0
    }

    # Track DDoS victims to compute average latency
    ddos_victims = []

    for line in dashboard_logs:
        # Count DDoS detections
        if "DDoS Attack detected on Node" in line:
            summary["ddos_detections"] += 1

        # Sybil summary lines
        m = re.search(r"Sybil Attack Summary: (\d+) sybil nodes added", line)
        if m:
            summary["sybil_nodes_added"] += int(m.group(1))

        # Concurrent lookups
        if "[Concurrent Lookup]" in line:
            if "Found" in line:
                summary["concurrent_found"] += 1
            else:
                summary["concurrent_not_found"] += 1

        # DDoS Attack Summary lines
        m = re.search(r"Victim Node (\d+) -> Attack Requests: (\d+), Total Latency: ([\d\.]+)s", line)
        if m:
            ddos_victims.append({
                "node_id": int(m.group(1)),
                "requests": int(m.group(2)),
                "latency": float(m.group(3))
            })

    # Compute ddos_victims_count, total_latency, avg_latency
    if ddos_victims:
        summary["ddos_victims_count"] = len(ddos_victims)
        tot_lat = sum(v["latency"] for v in ddos_victims)
        summary["ddos_total_latency"] = tot_lat
        summary["ddos_avg_latency"] = tot_lat / len(ddos_victims)

    # Compute total_lookups and success/failure rates
    total_concurrent = summary["concurrent_found"] + summary["concurrent_not_found"]
    summary["total_lookups"] = total_concurrent
    if total_concurrent > 0:
        summary["lookup_success_rate"] = (summary["concurrent_found"] / total_concurrent) * 100
        summary["lookup_failure_rate"] = (summary["concurrent_not_found"] / total_concurrent) * 100

    return summary

###############################################################################
# CONTINUOUS SIMULATION LOOP
###############################################################################
def continuous_simulation():
    global global_network
    log("\n🔹 Starting continuous Kademlia simulation...")
    net = Network()
    while len(net.nodes) < 1000:
        nid = random.randint(0, 2 ** BITSPACE - 1)
        if nid not in net.nodes:
            net.add_node(Node(nid))
    global_network = net
    log(f"Initial network size: {len(net.nodes)} nodes.")

    keys = ["fileA.txt", "fileB.mp4", "docC.pdf"]
    for k in keys:
        st_nodes, rounds = store_value(net, k, f"Data for {k}")
        log(f"Stored '{k}' on {len(st_nodes)} nodes (lookup rounds: {rounds})")

    # Continuous loop of traffic, attacks, and network churn.
    while True:
        simulate_normal_traffic(net, num_requests=500)
        attack_choice = random.choice(["ddos", "sybil", "poison", "eclipse"])
        if attack_choice == "ddos":
            # Increase attackers to 10 and victims to 100 for more aggressive DDoS
            simulate_ddos_attack(net, duration=10, num_attackers=10, num_victims=100, keys=keys)
        elif attack_choice == "sybil":
            simulate_sybil_attack(net, num_sybil=100)
        elif attack_choice == "poison":
            simulate_routing_table_poisoning(net, num_poisoners=20, duration=10, keys=keys)
        elif attack_choice == "eclipse":
            simulate_eclipse_attack(net, duration=10, num_eclipse_nodes=50, keys=keys)

        # Random node leaves / new node joins
        if random.random() < 0.3:
            leaving = net.get_random_node()
            log(f"* Node {leaving.node_id} is leaving the network.")
            net.remove_node(leaving.node_id)
            new_id = random.randint(0, 2 ** BITSPACE - 1)
            while new_id in net.nodes:
                new_id = random.randint(0, 2 ** BITSPACE - 1)
            new_node = Node(new_id)
            net.add_node(new_node)
            log(f"* New node {new_node.node_id} has joined the network.")

        log("\n-- Post-iteration check --")
        for k in keys:
            s_node = net.get_random_node()
            val, r = lookup_value(net, k, s_node)
            if val:
                log(f"Lookup for '{k}': Found from node {s_node.node_id} in {r} rounds.")
            else:
                log(f"Lookup for '{k}': Not found from node {s_node.node_id}.")
        time.sleep(5)


###############################################################################
# FLASK APP
###############################################################################
app = Flask(__name__)

@app.route("/")
def dashboard():
    return render_template("dashboard_nomit.html")

@app.route("/logs")
def logs():
    return jsonify({"logs": dashboard_logs})

@app.route("/concurrent")
def concurrent():
    return jsonify({"logs": concurrent_lookup_logs})

@app.route("/summary")
def summary():
    return jsonify({"summary": compute_summary()})

if __name__ == "__main__":
    sim_thread = threading.Thread(target=continuous_simulation, daemon=True)
    sim_thread.start()
    app.run(debug=True)
