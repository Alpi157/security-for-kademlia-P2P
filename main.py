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


dashboard_logs = []
concurrent_lookup_logs = []


def log(message):
    print(message)
    dashboard_logs.append(message)
    if "[Concurrent Lookup]" in message:
        concurrent_lookup_logs.append(message)



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

# For Sybil mitigation: allow at most 2 sybil nodes per bucket.
SYBIL_MAX_PER_BUCKET = 2

# Global mitigation thresholds (RL parameters)
DDOS_THRESHOLD_REQUESTS = 10  # if attack_requests > this, trigger mitigation
POISONING_FAILURE_THRESHOLD = 0.5  # threshold for failed pings in a bucket
ECLIPSE_DIVERSITY_THRESHOLD = 0.8  # if > threshold fraction from one /24, flag eclipse

# RL agent parameters for Q-learning
ACTION_SPACE = [
    "increase_ddos_mitigation",  # make DDoS mitigation more aggressive
    "decrease_ddos_mitigation",  # make DDoS mitigation less aggressive
    "tighten_sybil",  # lower SYBIL_MAX_PER_BUCKET
    "loosen_sybil",  # raise SYBIL_MAX_PER_BUCKET
    "tighten_poisoning",  # lower POISONING_FAILURE_THRESHOLD
    "loosen_poisoning",  # raise POISONING_FAILURE_THRESHOLD
    "tighten_eclipse",  # lower ECLIPSE_DIVERSITY_THRESHOLD
    "loosen_eclipse",  # raise ECLIPSE_DIVERSITY_THRESHOLD
    "rebalance_routing_table",  # force rebalancing of routing tables
    "trigger_eclipse_mitigation"  # trigger eclipse mitigation on a random node
]

# Q-learning hyperparameters
ALPHA_RL = 0.1  # learning rate
GAMMA = 0.9  # discount factor
EPSILON = 0.1  # exploration probability; will decay over time

# Q-table: maps from state (tuple) to a list of Q-values (one per action)
Q_table = {}

# For tracking previous state and action (for Q-learning update)
last_state = None
last_action = None
last_reward = None

# Global pointer to network (set later)
global_network = None

# Global cumulative reward for RL (accumulated over time)
cumulative_rl_reward = 0


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
        if features['latency'] > LATENCY_THRESHOLD or features['attack_requests'] > 5:
            return 1
        return 0


def print_detection(victim):
    log(f"DDoS Attack detected on Node {victim.node_id}")


###############################################################################
# NODE & NETWORK CLASSES (with Adaptive Mitigation for DDoS & Sybil)
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
        self.mitigation_active = False
        self.mitigation_factor = 1.0
        self.mitigation_activated_at = 0

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
        if contact.ip.startswith("10.0.0."):
            subnet = ".".join(contact.ip.split(".")[:3])
            count = sum(1 for c in bucket if ".".join(c.ip.split(".")[:3]) == subnet)
            if count >= SYBIL_MAX_PER_BUCKET:
                log(f"[Sybil Mitigation] Node {self.node_id}: Rejected contact {contact.node_id} from subnet {subnet} (count={count}).")
                return
        # Standard update
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
        return list({c.node_id: c for c in all_contacts}.values())

    def find_node(self, target_id):
        if self.is_poisoner:
            return []
        if self.mitigation_active:
            if random.random() < 0.5 * self.mitigation_factor:
                self.failed_lookups += 1
                return []
        if self.request_queue > CONGESTION_THRESHOLD:
            self.drop_rate = 0.5
            if random.random() < self.drop_rate:
                self.failed_lookups += 1
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
# ECLIPSE / POISONING DETECTION (UNCHANGED)
###############################################################################

def detect_eclipse_by_routing_diversity(network, threshold=ECLIPSE_DIVERSITY_THRESHOLD):
    flagged_nodes = []
    for node in network.all_nodes():
        for b_idx, bucket in node.buckets.items():
            if not bucket:
                continue
            subnets = {}
            for contact in bucket:
                subnet = ".".join(contact.ip.split(".")[:3])
                subnets[subnet] = subnets.get(subnet, 0) + 1
            mc_fraction = max(subnets.values()) / len(bucket)
            if mc_fraction >= threshold:
                flagged_nodes.append(node)
                log(f"[Eclipse Diversity Detection] Node {node.node_id} flagged: bucket {b_idx} has {mc_fraction * 100:.1f}% contacts from one subnet.")
                break
    if not flagged_nodes:
        log("[Eclipse Diversity Detection] No nodes flagged for low routing table diversity.")
    return flagged_nodes


def detect_poisoning_by_liveness(network, ping_attempts=3, bucket_failure_threshold=POISONING_FAILURE_THRESHOLD):
    flagged_nodes = []
    for node in network.all_nodes():
        for b_idx, bucket in node.buckets.items():
            if not bucket:
                continue
            failures = 0
            for contact in bucket:
                successes = sum(contact.ping() for _ in range(ping_attempts))
                if successes / ping_attempts < (1 - bucket_failure_threshold):
                    failures += 1
            if len(bucket) and (failures / len(bucket)) > bucket_failure_threshold:
                flagged_nodes.append(node)
                log(f"[Routing Poisoning Detection - Liveness] Node {node.node_id} flagged: {failures}/{len(bucket)} contacts in bucket {b_idx} failed liveness check.")
                break
    if not flagged_nodes:
        log("[Routing Poisoning Detection - Liveness] No nodes flagged based on liveness checks.")
    return flagged_nodes


def detect_poisoning_by_lookup_failures(network, threshold=0.3, min_reqs=50):
    flagged_nodes = []
    for node in network.all_nodes():
        if node.total_requests >= min_reqs:
            rate = node.failed_lookups / node.total_requests
            if rate > threshold:
                flagged_nodes.append(node)
                log(f"[Routing Poisoning Detection - Lookup Failures] Node {node.node_id} flagged: {node.failed_lookups}/{node.total_requests} (failure rate: {rate:.2f}).")
    if not flagged_nodes:
        log("[Routing Poisoning Detection - Lookup Failures] No nodes flagged based on lookup failure rates.")
    return flagged_nodes


def detect_routing_table_poisoning_attack(network):
    log("\n--- Running Routing Table Poisoning Detection ---")
    flagged_liveness = detect_poisoning_by_liveness(network)
    flagged_lookup = detect_poisoning_by_lookup_failures(network)
    flagged = set(flagged_liveness + flagged_lookup)
    if flagged:
        log(f"Total nodes flagged for potential routing table poisoning: {[n.node_id for n in flagged]}")
    else:
        log("No nodes flagged for routing table poisoning.")
    return flagged


###############################################################################
# ADAPTIVE MITIGATION FUNCTIONS
###############################################################################
def update_ddos_mitigation(victim):
    if victim.attack_requests > DDOS_THRESHOLD_REQUESTS and not victim.mitigation_active:
        victim.mitigation_active = True
        victim.mitigation_factor = 0.5
        victim.mitigation_activated_at = time.time()
        log(f"[Adaptive Mitigation] Activated on Node {victim.node_id}: Throttling incoming requests.")
    COOLDOWN = 10
    if victim.mitigation_active and time.time() - victim.mitigation_activated_at > COOLDOWN:
        victim.mitigation_active = False
        victim.mitigation_factor = 1.0
        log(f"[Adaptive Mitigation] Deactivated on Node {victim.node_id}: Recovery complete.")


def mitigate_sybil_attack(network):
    log("\n--- Running Sybil Attack Mitigation ---")
    mitigated_nodes = []
    for node in network.all_nodes():
        for b_idx, bucket in node.buckets.items():
            sybil_contacts = [c for c in bucket if c.is_sybil]
            if len(sybil_contacts) > SYBIL_MAX_PER_BUCKET:
                to_remove = sybil_contacts[SYBIL_MAX_PER_BUCKET:]
                for c in to_remove:
                    bucket.remove(c)
                mitigated_nodes.append(node.node_id)
                log(f"[Sybil Mitigation] Node {node.node_id}: Bucket {b_idx} sybil contacts reduced to {SYBIL_MAX_PER_BUCKET}.")
    if not mitigated_nodes:
        log("[Sybil Mitigation] No nodes required sybil mitigation.")
    return mitigated_nodes


def mitigate_routing_table_poisoning(network):
    log("\n--- Running Routing Table Poisoning Mitigation ---")
    mitigated_nodes = []
    for node in network.all_nodes():
        modified = False
        for b_idx, bucket in node.buckets.items():
            if not bucket:
                continue
            good_contacts = []
            removed = 0
            for contact in bucket:
                successes = sum(contact.ping() for _ in range(3))
                if successes / 3 >= (1 - POISONING_FAILURE_THRESHOLD):
                    good_contacts.append(contact)
                else:
                    removed += 1
            if removed > 0:
                additional_needed = BUCKET_SIZE - len(good_contacts)
                if additional_needed > 0:
                    candidates = [n for n in network.all_nodes() if
                                  n.node_id not in [c.node_id for c in good_contacts] and not n.is_poisoner]
                    candidates.sort(key=lambda n: abs(n.node_id - node.node_id))
                    good_contacts.extend(candidates[:additional_needed])
                node.buckets[b_idx] = good_contacts[:BUCKET_SIZE]
                log(f"[Routing Table Poisoning Mitigation] Node {node.node_id}: Bucket {b_idx} cleaned; removed {removed} unresponsive contacts.")
                modified = True
        if modified:
            mitigated_nodes.append(node.node_id)
    if not mitigated_nodes:
        log("[Routing Table Poisoning Mitigation] No nodes required mitigation.")
    return mitigated_nodes


def mitigate_eclipse_attack(network, victim, threshold=ECLIPSE_DIVERSITY_THRESHOLD):
    log(f"\n--- Running Eclipse Attack Mitigation on Node {victim.node_id} ---")
    for b_idx, bucket in victim.buckets.items():
        if not bucket:
            continue
        subnets = {}
        for contact in bucket:
            subnet = ".".join(contact.ip.split(".")[:3])
            subnets[subnet] = subnets.get(subnet, 0) + 1
        most_common_subnet = max(subnets, key=subnets.get)
        fraction = subnets[most_common_subnet] / len(bucket)
        if fraction >= threshold:
            new_bucket = [c for c in bucket if ".".join(c.ip.split(".")[:3]) != most_common_subnet]
            num_to_fill = BUCKET_SIZE - len(new_bucket)
            candidates = [n for n in network.all_nodes() if n.node_id not in [c.node_id for c in new_bucket]
                          and not n.is_eclipse and ".".join(n.ip.split(".")[:3]) != most_common_subnet]
            candidates.sort(key=lambda n: abs(n.node_id - victim.node_id))
            new_contacts = candidates[:num_to_fill]
            new_bucket.extend(new_contacts)
            victim.buckets[b_idx] = new_bucket[:BUCKET_SIZE]
            log(f"[Eclipse Mitigation] Node {victim.node_id}: Bucket {b_idx} mitigated; removed contacts from subnet {most_common_subnet} and refilled with {len(new_contacts)} new contacts.")


###############################################################################
# RL AGENT: STATE, ACTION, REWARD, AND Q-LEARNING UPDATE
###############################################################################
# Use coarser, bounded state discretization to keep state space small.
def discretize_state(state):
    return (
        min(state["ddos_detections"] // 5, 5),  # 0 to 5
        min(state["sybil_nodes_added"] // 50, 5),  # 0 to 5
        min(state["concurrent_not_found"] // 10, 5),  # 0 to 5
        round(state["ddos_avg_latency"] * 10),  # roughly 0-10
    )


# Updated reward function:
# Instead of heavily penalizing, we now give a bonus for a high success rate.
def compute_reward():
    state = compute_summary()
    total_concurrent = state["concurrent_found"] + state["concurrent_not_found"]
    success_rate = (state["concurrent_found"] / total_concurrent) if total_concurrent > 0 else 0
    latency_penalty = min(state["ddos_avg_latency"], 1.0)
    ddos_penalty = min(state["ddos_detections"] / 10.0, 1.0)
    reward = (success_rate * 100) + (state["concurrent_found"] * 0.1) - (latency_penalty * 30) - (ddos_penalty * 10)
    return reward, state


def choose_action(state):
    global Q_table, EPSILON
    if random.random() < EPSILON or state not in Q_table:
        return random.choice(ACTION_SPACE)
    else:
        q_values = Q_table[state]
        max_index = q_values.index(max(q_values))
        return ACTION_SPACE[max_index]


def update_q(state, action, reward, next_state):
    global Q_table, ALPHA_RL, GAMMA
    if state not in Q_table:
        Q_table[state] = [0.0 for _ in ACTION_SPACE]
    if next_state not in Q_table:
        Q_table[next_state] = [0.0 for _ in ACTION_SPACE]
    action_index = ACTION_SPACE.index(action)
    current_q = Q_table[state][action_index]
    max_next = max(Q_table[next_state])
    Q_table[state][action_index] = current_q + ALPHA_RL * (reward + GAMMA * max_next - current_q)


def compute_summary():
    global cumulative_rl_reward, global_network
    summary = {
        "ddos_detections": 0,
        "sybil_nodes_added": 0,
        "poisoned_entries": 0,
        "poisoned_total": 0,
        "eclipse_entries": 0,
        "eclipse_total": 0,
        "concurrent_found": 0,
        "concurrent_not_found": 0,
        "sybil_infiltrated": 0,
        "sybil_infiltration_total": 0,
        "sybil_infiltration_percent": 0.0,
        "ddos_victims_count": 0,
        "ddos_total_latency": 0.0,
        "ddos_avg_latency": 0.0,
        "eclipse_flagged_nodes": [],
        "poisoning_flagged_nodes": []
    }
    ddos_victims = []
    for line in dashboard_logs:
        if "DDoS Attack detected on Node" in line:
            summary["ddos_detections"] += 1
        m = re.search(r"Sybil Attack Summary: (\d+) sybil nodes added", line)
        if m:
            summary["sybil_nodes_added"] += int(m.group(1))
        m = re.search(r"Routing Table Poisoning Summary: (\d+) out of (\d+)", line)
        if m:
            summary["poisoned_entries"] = int(m.group(1))
            summary["poisoned_total"] = int(m.group(2))
        m = re.search(r"Eclipse Attack Summary on Node \d+: (\d+)\/(\d+)", line)
        if m:
            summary["eclipse_entries"] = int(m.group(1))
            summary["eclipse_total"] = int(m.group(2))
        if "[Concurrent Lookup]" in line:
            if "Found" in line:
                summary["concurrent_found"] += 1
            else:
                summary["concurrent_not_found"] += 1
        infil = re.search(r"Routing Table Infiltration: (\d+)/(\d+) entries \(([\d\.]+)%\)", line)
        if infil:
            summary["sybil_infiltrated"] = int(infil.group(1))
            summary["sybil_infiltration_total"] = int(infil.group(2))
            summary["sybil_infiltration_percent"] = float(infil.group(3))
        dv = re.search(r"Victim Node (\d+) -> Attack Requests: (\d+), Total Latency: ([\d\.]+)s", line)
        if dv:
            ddos_victims.append({
                "node_id": int(dv.group(1)),
                "requests": int(dv.group(2)),
                "latency": float(dv.group(3))
            })
        ef = re.search(r"\[Eclipse Diversity Detection\] Node (\d+) flagged", line)
        if ef:
            summary["eclipse_flagged_nodes"].append(int(ef.group(1)))
        pf = re.search(r"Node (\d+) flagged", line)
        if pf:
            summary["poisoning_flagged_nodes"].append(int(pf.group(1)))
    if ddos_victims:
        summary["ddos_victims_count"] = len(ddos_victims)
        tot_lat = sum(v["latency"] for v in ddos_victims)
        summary["ddos_total_latency"] = tot_lat
        summary["ddos_avg_latency"] = tot_lat / len(ddos_victims)

    # Additional metrics:
    total_concurrent = summary["concurrent_found"] + summary["concurrent_not_found"]
    summary["total_lookups"] = total_concurrent
    if total_concurrent > 0:
        summary["lookup_success_rate"] = summary["concurrent_found"] / total_concurrent * 100
        summary["lookup_failure_rate"] = summary["concurrent_not_found"] / total_concurrent * 100
    else:
        summary["lookup_success_rate"] = 0
        summary["lookup_failure_rate"] = 0

    if global_network:
        summary["network_size"] = len(global_network.nodes)
    else:
        summary["network_size"] = 0

    summary["cumulative_rl_reward"] = cumulative_rl_reward

    return summary


def run_rl_loop():
    global last_state, last_action, last_reward, global_network, EPSILON, cumulative_rl_reward
    last_state = discretize_state(compute_summary())
    last_action = None
    last_reward = 0
    while True:
        time.sleep(10)  # RL update interval
        reward, current_full_state = compute_reward()
        current_state = discretize_state(current_full_state)
        action = choose_action(last_state)
        apply_action(action)
        time.sleep(10)  # Allow time for action effects
        new_reward, new_full_state = compute_reward()
        new_state = discretize_state(new_full_state)
        update_q(last_state, action, new_reward, new_state)
        log(f"[RL] Action: {action}, New Reward: {new_reward:.2f}, Delta: {new_reward - last_reward:.2f}")
        log(f"[RL DEBUG] State: {last_state} Q-values: {Q_table.get(last_state, [])}")
        last_state = new_state
        last_reward = new_reward
        cumulative_rl_reward += new_reward
        # Slow epsilon decay a bit more (e.g., multiply by 0.998 per cycle)
        EPSILON = max(0.01, EPSILON * 0.998)


###############################################################################
# RL ACTIONS
###############################################################################
def apply_action(action):
    global DDOS_THRESHOLD_REQUESTS, SYBIL_MAX_PER_BUCKET, POISONING_FAILURE_THRESHOLD, ECLIPSE_DIVERSITY_THRESHOLD, global_network
    if action == "increase_ddos_mitigation":
        DDOS_THRESHOLD_REQUESTS = max(5, DDOS_THRESHOLD_REQUESTS - 1)
        log(f"[RL Action] Increased DDoS Mitigation: DDOS_THRESHOLD_REQUESTS set to {DDOS_THRESHOLD_REQUESTS}")
    elif action == "decrease_ddos_mitigation":
        DDOS_THRESHOLD_REQUESTS = min(20, DDOS_THRESHOLD_REQUESTS + 1)
        log(f"[RL Action] Decreased DDoS Mitigation: DDOS_THRESHOLD_REQUESTS set to {DDOS_THRESHOLD_REQUESTS}")
    elif action == "tighten_sybil":
        SYBIL_MAX_PER_BUCKET = max(1, SYBIL_MAX_PER_BUCKET - 1)
        log(f"[RL Action] Tightened Sybil Mitigation: SYBIL_MAX_PER_BUCKET set to {SYBIL_MAX_PER_BUCKET}")
    elif action == "loosen_sybil":
        SYBIL_MAX_PER_BUCKET = min(5, SYBIL_MAX_PER_BUCKET + 1)
        log(f"[RL Action] Loosened Sybil Mitigation: SYBIL_MAX_PER_BUCKET set to {SYBIL_MAX_PER_BUCKET}")
    elif action == "tighten_poisoning":
        POISONING_FAILURE_THRESHOLD = max(0.3, POISONING_FAILURE_THRESHOLD - 0.05)
        log(f"[RL Action] Tightened Poisoning Mitigation: POISONING_FAILURE_THRESHOLD set to {POISONING_FAILURE_THRESHOLD:.2f}")
    elif action == "loosen_poisoning":
        POISONING_FAILURE_THRESHOLD = min(0.7, POISONING_FAILURE_THRESHOLD + 0.05)
        log(f"[RL Action] Loosened Poisoning Mitigation: POISONING_FAILURE_THRESHOLD set to {POISONING_FAILURE_THRESHOLD:.2f}")
    elif action == "tighten_eclipse":
        ECLIPSE_DIVERSITY_THRESHOLD = max(0.6, ECLIPSE_DIVERSITY_THRESHOLD - 0.05)
        log(f"[RL Action] Tightened Eclipse Mitigation: ECLIPSE_DIVERSITY_THRESHOLD set to {ECLIPSE_DIVERSITY_THRESHOLD:.2f}")
    elif action == "loosen_eclipse":
        ECLIPSE_DIVERSITY_THRESHOLD = min(0.9, ECLIPSE_DIVERSITY_THRESHOLD + 0.05)
        log(f"[RL Action] Loosened Eclipse Mitigation: ECLIPSE_DIVERSITY_THRESHOLD set to {ECLIPSE_DIVERSITY_THRESHOLD:.2f}")
    elif action == "rebalance_routing_table":
        for node in global_network.all_nodes():
            for b_idx in range(BITSPACE):
                node.buckets[b_idx] = []
                contacts = random.sample(global_network.all_nodes(), min(BUCKET_SIZE, len(global_network.nodes)))
                for c in contacts:
                    if c.node_id != node.node_id:
                        node.update_routing_table(c)
        log("[RL Action] Rebalanced routing tables for all nodes.")
    elif action == "trigger_eclipse_mitigation":
        victim = global_network.get_random_node()
        mitigate_eclipse_attack(global_network, victim, threshold=ECLIPSE_DIVERSITY_THRESHOLD)
        log(f"[RL Action] Triggered eclipse mitigation on Node {victim.node_id}.")


###############################################################################
# SIMULATION FUNCTIONS
###############################################################################
def simulate_normal_traffic(network, num_requests=NUM_NORMAL_REQUESTS):
    for _ in range(num_requests):
        key = f"normal_file_{random.randint(1, 100)}.txt"
        start_node = network.get_random_node()
        _, _ = iterative_lookup(network, start_node, int(hashlib.sha1(key.encode()).hexdigest(), 16) % (2 ** BITSPACE))


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
    while time.time() - t0 < duration:
        for _ in range(5):
            attacker = random.choice(attackers)
            victim = random.choice(victims)
            random_key = "target_file.txt"
            t_id = int(hashlib.sha1(random_key.encode()).hexdigest(), 16) % (2 ** BITSPACE)
            s0 = time.time()
            _, _ = iterative_lookup(network, attacker, t_id)
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
            update_ddos_mitigation(victim)
        time.sleep(0.1)
    t.join()
    log("\nDDoS Attack Summary:")
    for v in victims:
        log(f"Victim Node {v.node_id} -> Attack Requests: {v.attack_requests}, Total Latency: {v.latency:.4f}s")
    for v in victims:
        if v.mitigation_active:
            v.mitigation_active = False
            v.mitigation_factor = 1.0
            log(f"[Adaptive Mitigation] Reset on Node {v.node_id}")


def simulate_sybil_attack(network, num_sybil=SYBIL_COUNT):
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
    mitigate_sybil_attack(network)


def simulate_routing_table_poisoning(network, num_poisoners=POISONER_COUNT, duration=POISONING_DURATION,
                                     keys=LOOKUP_KEYS):
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
    detect_routing_table_poisoning_attack(network)
    mitigate_routing_table_poisoning(network)


def simulate_eclipse_attack(network, duration=ECLIPSE_DURATION, num_eclipse_nodes=ECLIPSE_COUNT, keys=LOOKUP_KEYS):
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
    flagged = detect_eclipse_by_routing_diversity(network, threshold=ECLIPSE_DIVERSITY_THRESHOLD)
    if flagged:
        log(f"Nodes flagged for potential eclipse (low routing diversity): {[n.node_id for n in flagged]}")
    mitigate_eclipse_attack(network, victim, threshold=ECLIPSE_DIVERSITY_THRESHOLD)


###############################################################################
# CONTINUOUS SIMULATION LOOP (UPDATED)
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
            simulate_ddos_attack(net, duration=10, num_attackers=5, num_victims=50, keys=keys)
        elif attack_choice == "sybil":
            simulate_sybil_attack(net, num_sybil=50)
        elif attack_choice == "poison":
            simulate_routing_table_poisoning(net, num_poisoners=10, duration=10, keys=keys)
        elif attack_choice == "eclipse":
            simulate_eclipse_attack(net, duration=10, num_eclipse_nodes=30, keys=keys)
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



app = Flask(__name__)


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


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
    rl_thread = threading.Thread(target=run_rl_loop, daemon=True)
    rl_thread.start()
    sim_thread = threading.Thread(target=continuous_simulation, daemon=True)
    sim_thread.start()
    app.run(debug=True)
