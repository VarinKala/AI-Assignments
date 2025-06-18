import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import tracemalloc

# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.


# Algorithm: Iterative Deepening Search (IDS)

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def get_ids_path(adj_matrix, start_node, goal_node):
  tracemalloc.clear_traces()
  start_time = time.perf_counter()
  tracemalloc.start()

  path = [start_node]
  path_cost = [0]
  visited = []

  if start_node == goal_node:
    current, peak = tracemalloc.get_traced_memory()
    end_time = time.perf_counter()
    tracemalloc.stop()
    elapsed_time = end_time - start_time
    # return path, 0, elapsed_time, peak
    return path

  def DFS(cur_node, limit):
    if cur_node == goal_node:
      return True
    
    if limit == 0:
      return False
    
    visited.append(cur_node)
    
    for i in range(125):
      if adj_matrix[cur_node][i] > 0 and i not in visited:
        path.append(i)
        path_cost[0] += adj_matrix[cur_node][i]
        if DFS(i, limit - 1):
          return True
        path.pop()
        path_cost[0] -= adj_matrix[cur_node][i]
    
    return False

  for limit in range(10, 126, 10):
    visited = []
    if DFS(start_node, limit):
      current, peak = tracemalloc.get_traced_memory()
      end_time = time.perf_counter()
      tracemalloc.stop()
      elapsed_time = end_time - start_time
      # return path, path_cost[0], elapsed_time, peak
      return path
  
  current, peak = tracemalloc.get_traced_memory()
  end_time = time.perf_counter()
  tracemalloc.stop()
  elapsed_time = end_time - start_time
  # return None, -1, elapsed_time, peak
  return None


# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def get_bidirectional_search_path(adj_matrix, start_node, goal_node):
  tracemalloc.clear_traces()
  start_time = time.perf_counter()
  tracemalloc.start()

  if start_node == goal_node:
    current, peak = tracemalloc.get_traced_memory()
    end_time = time.perf_counter()
    tracemalloc.stop()
    elapsed_time = end_time - start_time
    # return [start_node], 0, elapsed_time, peak
    return [start_node]
 
  frontier_top = [start_node]
  frontier_bottom = [goal_node]

  parent_top = {start_node: None}
  parent_bottom = {goal_node: None}

  while frontier_top and frontier_bottom:
    top_node = frontier_top.pop(0)

    for i in range(125):
      if adj_matrix[top_node][i] > 0 and i not in parent_top:
        frontier_top.append(i)
        parent_top[i] = top_node

        if i in parent_bottom:
          path = [i]
          node = parent_top[i]
          if node:
            path_cost = adj_matrix[node][i]
          while node:
            path.append(node)
            if parent_top[node]:
              path_cost += adj_matrix[parent_top[node]][node]
            node = parent_top[node]
          
          path.reverse()

          node = parent_bottom[i]
          if node:
            path_cost += adj_matrix[i][node]
          while node:
            path.append(node)
            if parent_bottom[node]:
              path_cost += adj_matrix[node][parent_bottom[node]]
            node = parent_bottom[node]
          
          current, peak = tracemalloc.get_traced_memory()
          end_time = time.perf_counter()
          tracemalloc.stop()
          elapsed_time = end_time - start_time
          # return path, path_cost, elapsed_time, peak
          return path

    bottom_node = frontier_bottom.pop(0)

    for i in range(125):
      if adj_matrix[i][bottom_node] > 0 and i not in parent_bottom:
        frontier_bottom.append(i)
        parent_bottom[i] = bottom_node

        if i in parent_top:
          path = [i]
          node = parent_top[i]
          if node:
            path_cost = adj_matrix[node][i]
          while node:
            path.append(node)
            if parent_top[node]:
              path_cost += adj_matrix[parent_top[node]][node]
            node = parent_top[node]
          
          path.reverse()

          node = parent_bottom[i]
          if node:
            path_cost += adj_matrix[i][node]
          while node:
            path.append(node)
            if parent_bottom[node]:
              path_cost += adj_matrix[node][parent_bottom[node]]
            node = parent_bottom[node]
          
          current, peak = tracemalloc.get_traced_memory()
          end_time = time.perf_counter()
          tracemalloc.stop()
          elapsed_time = end_time - start_time
          # return path, path_cost, elapsed_time, peak
          return path

  current, peak = tracemalloc.get_traced_memory()
  end_time = time.perf_counter()
  tracemalloc.stop()
  elapsed_time = end_time - start_time
  # return None, -1, elapsed_time, peak
  return None


# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]

def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
  tracemalloc.clear_traces()
  start_time = time.perf_counter()
  tracemalloc.start()

  parent = {}
  frontier = {}
  sum_edge_weights = {start_node: 0}

  if start_node == goal_node:
    current, peak = tracemalloc.get_traced_memory()
    end_time = time.perf_counter()
    tracemalloc.stop()
    elapsed_time = end_time - start_time
    # return [start_node], 0, elapsed_time, peak
    return [start_node]

  def get_heuristic(node):
    start = node_attributes[start_node]
    cur = node_attributes[node]
    goal = node_attributes[goal_node]
    h_node = ((start['x'] - cur['x']) ** 2 + (start['y'] - cur['y']) ** 2) ** 0.5 + ((goal['x'] - cur['x']) ** 2 + (goal['y'] - cur['y']) ** 2) ** 0.5
    return h_node

  for i in range(125):
    if adj_matrix[start_node][i] > 0 and i != start_node:
      frontier[i] = adj_matrix[start_node][i] + get_heuristic(i)
      sum_edge_weights[i] = adj_matrix[start_node][i]
      parent[i] = start_node

  visited = [start_node]

  while frontier:
    min_i = start_node
    min_h = 1e9
    for i in frontier:
      if frontier[i] < min_h:
        min_i = i
        min_h = frontier[i]
    
    if min_i == goal_node:
      path = [goal_node]
      while True:
        if min_i == start_node:
          break

        min_i = parent[min_i]
        path.append(min_i)
      
      current, peak = tracemalloc.get_traced_memory()
      end_time = time.perf_counter()
      tracemalloc.stop()
      elapsed_time = end_time - start_time
      # return path[::-1], sum_edge_weights[goal_node], elapsed_time, peak
      return path[::-1]
    
    del frontier[min_i]
    visited.append(min_i)

    for i in range(125):
      if adj_matrix[min_i][i] > 0 and i not in visited:
        if i not in frontier or sum_edge_weights[min_i] + adj_matrix[min_i][i] + get_heuristic(i) < frontier[i]:
          sum_edge_weights[i] = sum_edge_weights[min_i] + adj_matrix[min_i][i]
          frontier[i] = sum_edge_weights[i] + get_heuristic(i)
          parent[i] = min_i

  current, peak = tracemalloc.get_traced_memory()
  end_time = time.perf_counter()
  tracemalloc.stop()
  elapsed_time = end_time - start_time
  # return None, -1, elapsed_time, peak
  return None


# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]

def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):
  tracemalloc.clear_traces()
  start_time = time.perf_counter()
  tracemalloc.start()

  if start_node == goal_node:
    current, peak = tracemalloc.get_traced_memory()
    end_time = time.perf_counter()
    tracemalloc.stop()
    elapsed_time = end_time - start_time
    # return [start_node], 0, elapsed_time, peak
    return [start_node]

  frontier_top = {start_node: 0}
  frontier_bottom = {goal_node: 0}

  parent_top = {start_node: None}
  parent_bottom = {goal_node: None}

  sum_edge_weights_top = {start_node: 0}
  sum_edge_weights_bottom = {goal_node: 0}

  def get_heuristic(node):
    start = node_attributes[start_node]
    cur = node_attributes[node]
    goal = node_attributes[goal_node]
    h_node = ((start['x'] - cur['x']) ** 2 + (start['y'] - cur['y']) ** 2) ** 0.5 + ((goal['x'] - cur['x']) ** 2 + (goal['y'] - cur['y']) ** 2) ** 0.5
    return h_node

  while frontier_top and frontier_bottom:
    best_top = None
    best_val = 1e9

    for i in frontier_top:
      if frontier_top[i] < best_val:
        best_top = i
        best_val = frontier_top[i]
      
    if best_top in parent_bottom:
      final_val = frontier_top[best_top] + frontier_bottom[best_top]
      for i in frontier_top:
        if i in frontier_bottom:
          if frontier_top[i] + frontier_bottom[i] < final_val:
            best_top = i
            final_val = frontier_top[i] + frontier_bottom[i]

      path = [best_top]
      node = parent_top[best_top]
      while node:
        path.append(node)
        node = parent_top[node] 
      
      path.reverse()

      node = parent_bottom[best_top]
      while node:
        path.append(node)
        node = parent_bottom[node]
      
      current, peak = tracemalloc.get_traced_memory()
      end_time = time.perf_counter()
      tracemalloc.stop()
      elapsed_time = end_time - start_time
      # return path, sum_edge_weights_bottom[best_top] + sum_edge_weights_top[best_top], elapsed_time, peak
      return path
    
    del frontier_top[best_top]
    
    for i in range(125):
      if adj_matrix[best_top][i] > 0 and i not in parent_top:
        if i not in frontier_top or sum_edge_weights_top[best_top] + adj_matrix[best_top][i] + get_heuristic(i) < frontier_top[i]:
          sum_edge_weights_top[i] = sum_edge_weights_top[best_top] + adj_matrix[best_top][i]
          frontier_top[i] = sum_edge_weights_top[i] + get_heuristic(i)
          parent_top[i] = best_top
    

    best_bottom = None
    best_val = 1e9

    for i in frontier_bottom:
      if frontier_bottom[i] < best_val:
        best_bottom = i
        best_val = frontier_bottom[i]
      
    if best_bottom in parent_top:
      final_val = frontier_top[best_bottom] + frontier_bottom[best_bottom]
      for i in frontier_bottom:
        if i in frontier_top:
          if frontier_top[i] + frontier_bottom[i] < final_val:
            best_bottom = i
            final_val = frontier_top[i] + frontier_bottom[i]

      path = [best_bottom]
      node = parent_top[best_bottom]
      while node:
        path.append(node)
        node = parent_top[node]
      
      path.reverse()

      node = parent_bottom[best_bottom]
      while node:
        path.append(node)
        node = parent_bottom[node]

      current, peak = tracemalloc.get_traced_memory()
      end_time = time.perf_counter()
      tracemalloc.stop()
      elapsed_time = end_time - start_time
      # return path, sum_edge_weights_top[best_bottom] + sum_edge_weights_bottom[best_bottom], elapsed_time, peak
      return path
    
    del frontier_bottom[best_bottom]
    
    for i in range(125):
      if adj_matrix[i][best_bottom] > 0 and i not in parent_bottom:
        if i not in frontier_bottom or sum_edge_weights_bottom[best_bottom] + adj_matrix[i][best_bottom] + get_heuristic(i) < frontier_bottom[i]:
          sum_edge_weights_bottom[i] = sum_edge_weights_bottom[best_bottom] + adj_matrix[i][best_bottom]
          frontier_bottom[i] = sum_edge_weights_bottom[i] + get_heuristic(i)
          parent_bottom[i] = best_bottom

  current, peak = tracemalloc.get_traced_memory()
  end_time = time.perf_counter()
  tracemalloc.stop()
  elapsed_time = end_time - start_time
  # return None, -1, elapsed_time, peak
  return None



# Bonus Problem
 
# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].

def bonus_problem(adj_matrix):
  vulnerable = []
  visited = []

  def DFS(source, vulnerable):
    visited[source] = True
    for i in range(125):
      if adj_matrix[source][i] > 0 and not visited[i]:
        DFS(i, vulnerable)
  
  for i in range(125):
    for j in range(125):
      if adj_matrix[i][j] > 0:
        visited = [False for _ in range(125)]
        edge_val1, edge_val2 = adj_matrix[i][j], adj_matrix[j][i]
        adj_matrix[i][j] = adj_matrix[j][i] = 0

        DFS(i, vulnerable)

        adj_matrix[i][j], adj_matrix[j][i] = edge_val1, edge_val2

        if not visited[j] and (min(i, j), max(i, j)) not in vulnerable:
          vulnerable.append((min(i, j), max(i, j)))
  
  return vulnerable

def obtain_all_paths_uninformed(adj_matrix):
  tracemalloc.clear_traces()

  start_time_1 = time.perf_counter()
  tracemalloc.start()
  for i in range(125):
    print(i)
    for j in range(125):
      get_ids_path(adj_matrix, i, j)
  
  current_1, peak_1 = tracemalloc.get_traced_memory()
  end_time_1 = time.perf_counter()
  tracemalloc.stop()
  tracemalloc.clear_traces()
  elapsed_time_1 = end_time_1 - start_time_1
  
  start_time_2 = time.perf_counter()
  tracemalloc.start()
  for i in range(125):
    print(i)
    for j in range(125):
      get_bidirectional_search_path(adj_matrix, i, j)

  current_2, peak_2 = tracemalloc.get_traced_memory()
  end_time_2 = time.perf_counter()
  tracemalloc.stop()
  elapsed_time_2 = end_time_2 - start_time_2

  print("\nTime taken for IDS:", elapsed_time_1, "seconds")
  print("Total Memory Used for IDS:", peak_1 / 10**6, "MB")
  print("\nTime taken for Bidirectional BFS:", elapsed_time_2, "seconds")
  print("Total Memory Used for Bidirectional BFS:", peak_2 / 10**6, "MB")

def obtain_all_paths_uninformed_2(adj_matrix):
  time_rec = [(0,0)]
  memory_rec = [(0,0)]

  for i in range(125):
    print(i, end = " ")
    for j in range(125):
      _, b, c, d = get_ids_path(adj_matrix, i, j)
      if b > 0:
        time_rec.append((b, c))
        memory_rec.append((b, d))

  time_rec_1 = np.array(time_rec)
  memory_rec_1 = np.array(memory_rec)
  print(time_rec_1, memory_rec_1)
  elapsed_time_1 = np.sum(time_rec_1[:, 1])
  peak_1 = np.sum(memory_rec_1[:, 1])
  

  time_rec = [(0,0)]
  memory_rec = [(0,0)]

  for i in range(125):
    print(i, end = " ")
    for j in range(125):
      _, b, c, d = get_bidirectional_search_path(adj_matrix, i, j)
      if b > 0:
        time_rec.append((b, c))
        memory_rec.append((b, d))

  time_rec_2 = np.array(time_rec)
  memory_rec_2 = np.array(memory_rec)
  elapsed_time_2 = np.sum(time_rec_2[:, 1])
  peak_2 = np.sum(memory_rec_2[:, 1])


  print("\nTime taken for IDS:", elapsed_time_1, "seconds")
  print("Total Memory Used for IDS:", peak_1 / 10**6, "MB")
  print("\nTime taken for Bidirectional BFS:", elapsed_time_2, "seconds")
  print("Total Memory Used for Bidirectional BFS:", peak_2 / 10**6, "MB")

  plt.scatter(time_rec_1[:, 0], time_rec_1[:, 1])
  plt.xlabel("Path Cost")
  plt.ylabel("Time Taken")
  plt.title("IDS")
  plt.show()

  plt.scatter(memory_rec_1[:, 0], memory_rec_1[:, 1])
  plt.xlabel("Path Cost")
  plt.ylabel("Memory")
  plt.title("IDS")
  plt.show()

  plt.scatter(time_rec_2[:, 0], time_rec_2[:, 1])
  plt.xlabel("Path Cost")
  plt.ylabel("Time Taken")
  plt.title("Bidirectional BFS")
  plt.show()

  plt.scatter(memory_rec_2[:, 0], memory_rec_2[:, 1])
  plt.xlabel("Path Cost")
  plt.ylabel("Memory")
  plt.title("Bidirectional BFS")
  plt.show()


def obtain_all_paths_informed(adj_matrix, node_attributes):
  tracemalloc.clear_traces()

  start_time_1 = time.perf_counter()
  tracemalloc.start()
  for i in range(125):
    print(i)
    for j in range(125):
      get_astar_search_path(adj_matrix, node_attributes, i, j)
  
  current_1, peak_1 = tracemalloc.get_traced_memory()
  end_time_1 = time.perf_counter()
  tracemalloc.stop()
  tracemalloc.clear_traces()
  elapsed_time_1 = end_time_1 - start_time_1
  
  start_time_2 = time.perf_counter()
  tracemalloc.start()
  for i in range(125):
    print(i)
    for j in range(125):
      get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, i, j)

  current_2, peak_2 = tracemalloc.get_traced_memory()
  end_time_2 = time.perf_counter()
  tracemalloc.stop()
  elapsed_time_2 = end_time_2 - start_time_2

  print("\nTime taken for A* Search:", elapsed_time_1, "seconds")
  print("Total Memory Used for A* Search:", peak_1 / 10**6, "MB")
  print("\nTime taken for Bidirectional A* Search:", elapsed_time_2, "seconds")
  print("Total Memory Used for Bidirectional A* Search:", peak_2 / 10**6, "MB")

def obtain_all_paths_informed_2(adj_matrix, node_attributes):
  time_rec = [(0,0)]
  memory_rec = [(0,0)]
  
  for i in range(125):
    print(i, end = " ")
    for j in range(125):
      _, b, c, d = get_astar_search_path(adj_matrix, node_attributes, i, j)
      if b > 0:
        time_rec.append((b, c))
        memory_rec.append((b, d))
  
  time_rec_1 = np.array(time_rec)
  memory_rec_1 = np.array(memory_rec)
  elapsed_time_1 = np.sum(time_rec_1[:, 1])
  peak_1 = np.sum(memory_rec_1[:, 1])
  

  time_rec = [(0,0)]
  memory_rec = [(0,0)]
  
  for i in range(125):
    print(i, end = " ")
    for j in range(125):
      _, b, c, d = get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, i, j)
      if b > 0:
        time_rec.append((b, c))
        memory_rec.append((b, d))

  time_rec_2 = np.array(time_rec)
  memory_rec_2 = np.array(memory_rec)
  elapsed_time_2 = np.sum(time_rec_2[:, 1])
  peak_2 = np.sum(memory_rec_2[:, 1])

  print("\nTime taken for A* Search:", elapsed_time_1, "seconds")
  print("Total Memory Used for A* Search:", peak_1 / 10**6, "MB")
  print("\nTime taken for Bidirectional A* Search:", elapsed_time_2, "seconds")
  print("Total Memory Used for Bidirectional A* Search:", peak_2 / 10**6, "MB")

  plt.scatter(time_rec_1[:, 0], time_rec_1[:, 1])
  plt.xlabel("Path Cost")
  plt.ylabel("Time Taken")
  plt.title("A*")
  plt.show()

  plt.scatter(memory_rec_1[:, 0], memory_rec_1[:, 1])
  plt.xlabel("Path Cost")
  plt.ylabel("Memory")
  plt.title("A*")
  plt.show()

  plt.scatter(time_rec_2[:, 0], time_rec_2[:, 1])
  plt.xlabel("Path Cost")
  plt.ylabel("Time Taken")
  plt.title("Bidirectional A* Search")
  plt.show()

  plt.scatter(memory_rec_2[:, 0], memory_rec_2[:, 1])
  plt.xlabel("Path Cost")
  plt.ylabel("Memory")
  plt.title("Bidirectional A* Search")
  plt.show()

  return time_rec_1, memory_rec_1, time_rec_2, memory_rec_2

# Time taken for A* Search: 87.53169724997133
# Total Memory Used for A* Search: 0.020628

# Time taken for Bidirectional A* Search: 70.6928527909331
# Total Memory Used for Bidirectional A* Search: 0.018591


if __name__ == "__main__":
  # adj_matrix = np.load('/Users/manojk/Desktop/IIIT Delhi/05 CSE643 AI/A1/Search_2022561/IIIT_Delhi.npy')
  adj_matrix = np.load('IIIT_Delhi.npy')
  # with open('/Users/manojk/Desktop/IIIT Delhi/05 CSE643 AI/A1/Search_2022561/IIIT_Delhi.pkl', 'rb') as f:
  with open('IIIT_Delhi.pkl', 'rb') as f:
    node_attributes = pickle.load(f)

  start_node = int(input("Enter the start node: "))
  end_node = int(input("Enter the end node: "))

  print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
  print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
  print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bonus Problem: {bonus_problem(adj_matrix)}')

  # obtain_all_paths_uninformed_2(adj_matrix)
  # print("\n")
  # obtain_all_paths_informed_2(adj_matrix, node_attributes)