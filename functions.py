import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pyqubo import Constraint, Placeholder, Binary
from itertools import groupby
import traci


def extract_lanes_from_state(net_path):
    # Parse the XML file
    tree = ET.parse(net_path)
    root = tree.getroot()

    tllogic_list = []
    connection_list = []

    # Loop through all the elements in the XML
    for elem in root:
        if elem.tag == "tlLogic":
            tllogic_id = elem.attrib['id']
            for phase in elem:
                phase_dict = phase.attrib
                phase_dict['tl_id'] = tllogic_id
                tllogic_list.append(phase_dict)
        elif elem.tag == "connection":
            connection_list.append(elem.attrib)

    # Filter the connection list to keep only entries with 'tl' and 'linkIndex'
    connection_list_filtered = [entry for entry in connection_list if 'tl' in entry and 'linkIndex' in entry]

    # Sort the filtered connection list by 'tl' and 'linkIndex' for grouping
    connection_list_sorted = sorted(connection_list_filtered, key=lambda x: (x['tl'], int(x['linkIndex'])))

    # Group the sorted connection list by 'tl' and 'linkIndex' and create a dictionary mapping from 'tl' to a list of 'lane'
    tl_to_lane = {tl: [f"{item['from']}_{item['fromLane']}" for item in group]
                  for tl, group in groupby(connection_list_sorted, key=lambda x: x['tl'])}

    state_to_lane = {}

    # Iterate over the tllogic_list
    for tllogic in tllogic_list:
        tl_id = tllogic['tl_id']
        if tl_id in tl_to_lane:
            lanes = tl_to_lane[tl_id]
            state = tllogic['state']
            # Create a dictionary mapping from each index of 'state' to the corresponding 'lane'
            state_to_lane[(tl_id, state)] = {i: lanes[i] for i in range(len(state))}

    lane_for_G_g = {}
    phase_counter = {}

    # Iterate over the state_to_lane dictionary
    for (tl_id, state), lane_dict in state_to_lane.items():
        # Get the lanes for 'G' or 'g'
        lanes_for_G_g = [lane for i, lane in lane_dict.items() if state[i] in ['G', 'g']]
        # Remove duplicates
        lanes_for_G_g_unique = list(set(lanes_for_G_g))
        
        if tl_id not in lane_for_G_g:
            lane_for_G_g[tl_id] = {}
            phase_counter[tl_id] = 0  # Initialize the phase counter for the tl_id
        
        # Add the lanes for 'G' or 'g' to the result dictionary
        lane_for_G_g[tl_id][f"phase{phase_counter[tl_id]}"] = lanes_for_G_g_unique
        phase_counter[tl_id] += 1  # Increment the phase counter

    return lane_for_G_g


def count_vehicles(lane_for_G_g, lam, net_path):
    vehicle_counts = {}
    connection_list = []

    # Parse the XML file again
    tree = ET.parse(net_path)
    root = tree.getroot()

    # Loop through all the elements in the XML
    for elem in root:
        if elem.tag == "connection":
            connection_list.append(elem.attrib)

    # Filter the connection list to keep only entries with 'tl' and 'linkIndex'
    connection_list_filtered = [entry for entry in connection_list if 'tl' in entry and 'linkIndex' in entry]

    # Sort the filtered connection list by 'tl' and 'linkIndex' for grouping
    connection_list_sorted = sorted(connection_list_filtered, key=lambda x: (x['tl'], int(x['linkIndex'])))

    # Group the sorted connection list by 'tl' and 'linkIndex' and create a dictionary mapping from 'tl' to a list of 'lane'
    tl_to_lane_dir = {tl: {f"{item['from']}_{item['fromLane']}": item['dir'] for item in group}
                      for tl, group in groupby(connection_list_sorted, key=lambda x: x['tl'])}

    # Iterate over the lane_for_G_g dictionary
    for tl_id, phase_dict in lane_for_G_g.items():
        if tl_id not in vehicle_counts:
            vehicle_counts[tl_id] = {}

        # Iterate over the phase_dict
        for phase, lanes in phase_dict.items():
            count = 0

            # Check if all lanes in this phase have 'dir' = 'r'
            all_lanes_right = all(tl_to_lane_dir[tl_id][lane] == 'r' for lane in lanes)

            # Iterate over the lanes
            for lane in lanes:
                vehicle_count = traci.lane.getLastStepVehicleNumber(lane)  # Count the vehicles in the lane
                if all_lanes_right:
                    vehicle_count *= lam  # Multiply the vehicle count by lam if all lanes have 'dir' = 'r'
                count += vehicle_count

            # Add the vehicle count to the result dictionary
            vehicle_counts[tl_id][phase] = count

    return vehicle_counts

def extract_edge_info(net_path):
    # Parse the XML file
    tree = ET.parse(net_path)
    root = tree.getroot()

    # Identify junctions with traffic lights
    junctions_with_lights = set()
    for junction in root.iter('junction'):
        if junction.attrib.get('type') == 'traffic_light':
            junctions_with_lights.add(junction.attrib.get('id'))

    edge_info = {}

    # Loop through all the elements in the XML
    for elem in root:
        if elem.tag == "edge" and 'from' in elem.attrib and 'to' in elem.attrib:
            from_node = elem.attrib['from']
            to_node = elem.attrib['to']
            
            # Only consider edges where both nodes are junctions with traffic lights
            if from_node in junctions_with_lights and to_node in junctions_with_lights:
                lengths = []
                speeds = []
                
                # Loop through all the lanes in the edge
                for lane in elem.iter('lane'):
                    length = float(lane.attrib.get('length', 0))  # Use 0 as default value
                    speed = float(lane.attrib.get('speed', 0))  # Use 0 as default value
                    lengths.append(length)
                    speeds.append(speed)

                # Use the average length and speed as representative values for the edge
                avg_length = sum(lengths) / len(lengths) if lengths else 0.0
                avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
                
                # Create a tuple for the nodes, ensuring it is in a consistent order
                node_tuple = tuple(sorted([from_node, to_node]))
                
                edge_info[node_tuple] = {"length": avg_length, "speed": avg_speed}

    return edge_info

def calculate_total_waiting_time(file_path):
    # XMLファイルを読み込む
    tree = ET.parse(file_path)
    root = tree.getroot()

    # waitingTimeの合計を初期化
    total_waiting_time = 0.0

    # すべての<tripinfo>タグをループ
    for tripinfo in root.findall('tripinfo'):
        # waitingTime属性を取得し、floatに変換
        waiting_time = float(tripinfo.get('waitingTime'))
        # 合計に加える
        total_waiting_time += waiting_time

    return total_waiting_time


def transition_state(str1, str2):
    # 文字列の長さが等しいことを確認
    assert len(str1) == len(str2), "The lengths of the two strings are not equal."

    # 新しい状態を生成
    new_str1 = ""
    for s1, s2 in zip(str1, str2):
        if (s1 in ['G', 'g'] and s2 == 'r'):
            new_str1 += 'y'
        else:
            new_str1 += s1

    return new_str1

def get_junc_info(sumocfg_path):
    '''
    交差点の情報を取得する
    '''
    sumoBinary = "PATH_YOUR_SUMO"
    sumoCmd = [sumoBinary, "-c", sumocfg_path, "--no-warnings", "--log", "sumo.log"]
    traci.start(sumoCmd)

    # ネットワーク内のすべての信号機のIDを取得します。
    traffic_lights = traci.trafficlight.getIDList()

    # 各信号機のフェーズ数を取得します。
    junc_info={}
    for tl in traffic_lights:
        # フェーズ定義を取得します。
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)

        # フェーズ数を取得します。
        num_phases = len(logic[0].phases)
        junc_info[tl] = {'num_modes': num_phases}
        print(f'Traffic light {tl} has {num_phases} phases.')

    traci.close()
    return junc_info


# import xml.etree.ElementTree as ET

def get_incLanes_from_net_file(net_path):
    # XMLファイルをパースします。
    tree = ET.parse(net_path)
    root = tree.getroot()

    # 交差点IDと入力レーンを格納する辞書を初期化します。
    junction_incoming_lanes = {}

    # すべての交差点要素に対してループします。
    for junction in root.iter('junction'):
        # 交差点のIDを取得します。
        junc_id = junction.get('id')

        # 交差点の入力レーンを取得します。
        incLanes = junction.get('incLanes')

        # 入力レーンを空白で分割してリストにします。
        incLanes_list = incLanes.split()

        # 交差点IDと入力レーンのリストを辞書に追加します。
        junction_incoming_lanes[junc_id] = incLanes_list # 南、東、北、西の順でレーンIDが格納されている

    return junction_incoming_lanes


def make_C(vehicle_counts, junc_info):
    C = {}
    for junc in junc_info.keys():
        for m in range(junc_info[junc]['num_modes']):
            phase = 'phase'+str(m)
            C[junc, m] = vehicle_counts[junc][phase]
    return C

def make_normalized_C(vehicle_counts, junc_info):
    C = {}
    max_value = 0
    # compute C and track the maximum value
    for junc in junc_info.keys():
        for m in range(junc_info[junc]['num_modes']):
            phase = 'phase' + str(m)
            C[junc, m] = vehicle_counts[junc][phase]
            max_value = max(max_value, C[junc, m])
    # normalize C by dividing each entry by the maximum value
    if max_value > 0:  # Avoid division by zero
        for key in C.keys():
            C[key] /= max_value
    return C


def make_B(edge_info):
    B = {}
    for (junc1, junc2), info in edge_info.items():
        B[junc1, junc2] = 1 / (info['length'] / info['speed'])
    return B

def make_R(edge_info, junc_info):
    R = {}
    for (junc1, junc2) in edge_info.keys():
        for m in range(junc_info[junc1]['num_modes']):
            for n in range(junc_info[junc2]['num_modes']):
                R[junc1, m, junc2, n] = 0 # 適当に0とする
    return R

def make_BR(B, R):
    BR = {}
    for junc1, m, junc2, n in R.keys():
        BR[junc1, m, junc2, n] = B[junc1, junc2] * R[junc1, m, junc2, n]
    return BR

def make_normalized_BR(B, R):
    BR = {}
    max_value = 0
    # compute BR and track the maximum value
    for junc1, m, junc2, n in R.keys():
        BR[junc1, m, junc2, n] = B[junc1, junc2] * R[junc1, m, junc2, n]
        max_value = max(max_value, BR[junc1, m, junc2, n])
    # normalize BR by dividing each entry by the maximum value
    if max_value > 0:  # Avoid division by zero
        for key in BR.keys():
            BR[key] /= max_value
    return BR

def convert_sol(dictionary):    
    new_dict = {}

    # Loop over the original dictionary
    for key, value in dictionary.items():
        # Extract the 'i' and 'm' parts from the key
        i_part = key.split('[')[1].rstrip(']')
        m_part = int(key.split('[')[-1].rstrip(']'))

        # If 'i' part is not in the new dictionary, add it
        if i_part not in new_dict:
            new_dict[i_part] = {}

        # Add the 'm' part and its corresponding value to the 'i' part of the new dictionary
        new_dict[i_part][m_part] = value

    return new_dict

def post_processing(dictionary, junc_info):
    '''
    制約のチェック
    もし満たしていない場合、満たすように後処理する
    '''
    for junc in junc_info.keys():

        if np.sum(list(dictionary[junc].values())) != 1:
            keys_with_value_one = [key for key, value in dictionary[junc].items() if value == 1]
            random_key = random.choice(keys_with_value_one)
            dictionary[junc] = {m: 0 for m in range(junc_info[junc]['num_modes'])}
            dictionary[junc][random_key] = 1

        if np.sum(list(dictionary[junc].values())) == 0:
            random_key = random.choice(list(dictionary[junc].keys()))
            dictionary[junc] = {m: 0 for m in range(junc_info[junc]['num_modes'])}
            dictionary[junc][random_key] = 1

    return dictionary

def get_mode(dictionary):
    mode_dict = {}
    for junc, sample_dict in dictionary.items():
        for key, val in sample_dict.items():
            if val == 1:
                mode_dict[junc] = key
    return mode_dict


def use_annealing(C, BR, alpha, beta, gamma, num_reads, sampler, junc_info):
    # バイナリ変数の作成
    x = {(i, m): Binary(f"x[{i}][{m}]") for i, m in C.keys()}

    # 目的関数の第1項
    H1 = np.sum([C[i, m] * x[i, m] for i, m in C.keys()])

    # 目的関数の第2項
    H2 = np.sum([BR[i, u, j, v] * x[i, u] * x[j, v] for i, u, j, v in BR.keys()])

    # 目的関数の第3項
    H3 = np.sum([(np.sum([x[i, m] for m in range(junc_info[i]['num_modes'])]) - 1) ** 2 \
                for i in junc_info.keys()])

    # 最小化したい目的関数
    H = (
        -Placeholder("alpha") * H1
        - Placeholder("beta") * H2
        + Placeholder("gamma") * Constraint(H3, "H3")
    )

    model = H.compile()
    feed_dict = {"alpha": alpha, "beta": beta, "gamma": gamma}
    qubo, offset = model.to_qubo(feed_dict=feed_dict)

    start_time = time.time()
    sampleset = sampler.sample_qubo(qubo, num_reads=num_reads)
    end_time = time.time()
    elapsed_time = end_time - start_time

    lowest_sample = sampleset.first.sample
    # 制約チェック
    lowest_dict = convert_sol(lowest_sample)
    lowest_dict = post_processing(lowest_dict, junc_info)

    return lowest_dict, sampleset, elapsed_time

def convert_sol_like_sa(sol):
    '''
    Gurobiの出力をSAの出力形式に合わせるため
    '''
    new_sol = {}
    for key, value in sol.items():
        key = key.replace(',', '][')
        new_sol[key] = value
    return new_sol

from gurobipy import *

def use_gurobi(C, BR, alpha, beta, gamma, junc_info):
    # Create a new Gurobi model
    model = Model("myModel")
    model.setParam('OutputFlag', 0)

    # Create binary variables
    x = model.addVars(C.keys(), vtype=GRB.BINARY, name="x")

    # Set objective function
    # Objective function's first term
    H1 = quicksum([C[i, m] * x[i, m] for i, m in C.keys()])

    # Objective function's second term
    H2 = quicksum([BR[i, u, j, v] * x[i, u] * x[j, v] for i, u, j, v in BR.keys()])

    # Objective function's third term
    H3 = quicksum([(quicksum([x[i, m] for m in range(junc_info[i]['num_modes'])]) - 1) ** 2 \
                for i in junc_info.keys()])

    # Set the objective to minimize
    model.setObjective(-alpha * H1 - beta * H2 + gamma * H3, GRB.MINIMIZE)

    # Optimize the model
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Return the solution
    sol = {v.varName: int(v.x) for v in model.getVars()}
    new_sol = convert_sol_like_sa(sol)

    # 制約チェック
    sol_dict = convert_sol(new_sol)
    sol_dict = post_processing(sol_dict, junc_info)

    return sol_dict, model.runtime, elapsed_time

import matplotlib.patches as mpatches


def show_transition(mode_log, figsize, annot=True, square=True, rotation=0):
    time_points = sorted(mode_log.keys())
    # Convert dictionary to DataFrame
    df = pd.DataFrame(mode_log)
    
    # Find the maximum value in the mode_log to determine the number of colors needed
    max_value = max(max(sub_dict.values()) for sub_dict in mode_log.values())
    
    # Create a color map with enough colors for each unique value
    cmap = plt.get_cmap("tab10", max_value+1)

    plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df, cmap=cmap, annot=annot, cbar=False, square=square)

    # Set x-ticks at representative values
    heatmap.set_xticks(range(0, max(time_points) + 10, 10))
    heatmap.set_xticklabels(range(0, max(time_points) + 10, 10), rotation=rotation)

    # Create legend
    patches = []
    for i in range(max_value+1):
        patch = mpatches.Patch(color=cmap(i), label=f'Mode{i+1}')
        patches.append(patch)

    plt.legend(handles=patches)

    plt.title("Mode Transition")
    plt.xlabel("Time Step [s]")
    plt.ylabel("Intersection")
    plt.show()




def convert_mode_log(mode_log):
    # Create new dictionary with filled intervals
    new_data_dict = {}
    time_points = sorted(mode_log.keys())

    for i, time_point in enumerate(time_points[:-1]):
        next_time_point = time_points[i+1]
        for time in range(time_point, next_time_point):
            new_data_dict[time] = mode_log[time_point]

    # Add the last interval
    for time in range(time_points[-1], time_points[-1] + 10):
        new_data_dict[time] = mode_log[time_points[-1]]

    return new_data_dict

def get_junc_info(sumocfg_path):
    '''
    交差点の情報を取得する
    '''
    sumoBinary = "PATH_YOUR_SUMO"
    sumoCmd = [sumoBinary, "-c", sumocfg_path, "--no-warnings", "--log", "sumo.log"]
    traci.start(sumoCmd)

    # ネットワーク内のすべての信号機のIDを取得します。
    traffic_lights = traci.trafficlight.getIDList()

    # 各信号機のフェーズ数を取得します。
    junc_info={}
    for tl in traffic_lights:
        # フェーズ定義を取得します。
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)

        # フェーズ数を取得します。
        num_phases = len(logic[0].phases)
        junc_info[tl] = {'num_modes': num_phases}
        print(f'Traffic light {tl} has {num_phases} phases.')

    traci.close()
    return junc_info

def get_info(net_path, sumocfg_path):
    '''
    交差点の情報を取得する
    '''
    sumoBinary = "PATH_YOUR_SUMO"
    sumoCmd = [sumoBinary, "-c", sumocfg_path, "--no-warnings", "--log", "sumo.log"]
    traci.start(sumoCmd)

    # ネットワーク内のすべての信号機のIDを取得します。
    traffic_lights = traci.trafficlight.getIDList()

    # 各信号機のフェーズ数を取得します。
    junc_info={}
    for tl in traffic_lights:
        # フェーズ定義を取得します。
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)

        # フェーズ数を取得します。
        num_phases = len(logic[0].phases)
        junc_info[tl] = {'num_modes': num_phases}
        print(f'Traffic light {tl} has {num_phases} phases.')

    traci.close()


    # Parse the XML file
    tree = ET.parse(net_path)
    root = tree.getroot()

    edge_info = {}

    # Loop through all the elements in the XML
    for elem in root:
        if elem.tag == "edge" and 'from' in elem.attrib and 'to' in elem.attrib:
            from_node = elem.attrib['from']
            to_node = elem.attrib['to']
            
            # Only consider edges where both nodes are junctions with traffic lights
            if from_node in traffic_lights and to_node in traffic_lights:
                lengths = []
                speeds = []
                
                # Loop through all the lanes in the edge
                for lane in elem.iter('lane'):
                    length = float(lane.attrib.get('length', 0))  # Use 0 as default value
                    speed = float(lane.attrib.get('speed', 0))  # Use 0 as default value
                    lengths.append(length)
                    speeds.append(speed)

                # Use the average length and speed as representative values for the edge
                avg_length = sum(lengths) / len(lengths) if lengths else 0.0
                avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
                
                # Create a tuple for the nodes, ensuring it is in a consistent order
                node_tuple = tuple(sorted([from_node, to_node]))
                
                edge_info[node_tuple] = {"length": avg_length, "speed": avg_speed}

    return junc_info, edge_info
