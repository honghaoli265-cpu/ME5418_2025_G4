#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# dstar_lite.py
import numpy as np
import heapq

class Node:
    def __init__(self, pos):
        self.pos = pos
        self.g = float('inf')
        self.rhs = float('inf')
        self.h = 0

    def __lt__(self, other):
        # 用于堆排序
        return (min(self.g, self.rhs) + self.h) < (min(other.g, other.rhs) + other.h)

class DStarLite:
    def __init__(self, start, goal, grid):
        """
        D* Lite 算法（3D 网格版本）
        start, goal: 三维网格坐标元组 (x, y, z)
        grid: 0 可行，1 障碍
        """
        self.start = start
        self.goal = goal
        self.grid = grid
        self.nodes = {}
        self.U = []  # 优先队列
        self.km = 0

        # 初始化节点
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                for z in range(grid.shape[2]):
                    self.nodes[(x, y, z)] = Node((x, y, z))

        self.nodes[goal].rhs = 0
        self.nodes[goal].h = self.heuristic(start, goal)
        heapq.heappush(self.U, (self.key(self.nodes[goal]), self.nodes[goal]))

    def heuristic(self, a, b):
        """欧式距离启发式"""
        return np.linalg.norm(np.array(a) - np.array(b))

    def key(self, n):
        k1 = min(n.g, n.rhs) + self.heuristic(self.start, n.pos) + self.km
        k2 = min(n.g, n.rhs)
        return (k1, k2)

    def get_neighbors(self, u):
        """6邻域搜索"""
        neighbors = []
        x, y, z = u.pos
        for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            nx, ny, nz = x+dx, y+dy, z+dz
            if 0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1] and 0 <= nz < self.grid.shape[2]:
                if self.grid[nx, ny, nz] == 0:
                    neighbors.append(self.nodes[(nx, ny, nz)])
        return neighbors

    def cost(self, a, b):
        return np.linalg.norm(np.array(a.pos) - np.array(b.pos))

    def update_vertex(self, u):
        """更新节点值"""
        if u.pos != self.goal:
            nbrs = self.get_neighbors(u)
            if nbrs:
                u.rhs = min([self.cost(u, s) + s.g for s in nbrs])
        # 清理旧项
        self.U = [(k, node) for k, node in self.U if node != u]
        heapq.heapify(self.U)
        # 重新入堆
        if u.g != u.rhs:
            heapq.heappush(self.U, (self.key(u), u))

    def compute_shortest_path(self):
        """主循环"""
        while self.U:
            k_old, u = heapq.heappop(self.U)
            if u.g > u.rhs:
                u.g = u.rhs
                for s in self.get_neighbors(u):
                    self.update_vertex(s)
            else:
                u.g = float('inf')
                self.update_vertex(u)
                for s in self.get_neighbors(u):
                    self.update_vertex(s)

    def get_path(self):
        """根据g值回溯路径"""
        path = []
        current = self.nodes[self.start]
        path.append(current.pos)
        while current.pos != self.goal:
            neighbors = self.get_neighbors(current)
            if not neighbors:
                break
            current = min(neighbors, key=lambda n: n.g + self.cost(current, n))
            path.append(current.pos)
        return path

