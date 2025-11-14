from collections import deque

def bfs_coloring(n, adj):
    color = [-1] * n
    q = deque()

    for start in range(n):
        if color[start] != -1:
            continue

        q.append(start)

        while q:
            u = q.popleft()
            used = set()
            for v in adj[u]:
                if color[v] >= 0:
                    used.add(color[v])

            c = 0
            while c in used:
                c += 1
            color[u] = c

            for v in adj[u]:
                if color[v] == -1:
                    color[v] = -2
                    q.append(v)

    return color


def bfs_coloring_layer_sorted(n, adj):
    color = [-1] * n
    visited = [False] * n

    for start in range(n):
        if visited[start]:
            continue

        q = deque([start])
        visited[start] = True

        while q:
            level_size = len(q)
            next_level = []

            for _ in range(level_size):
                u = q.popleft()

                used = set()
                for v in adj[u]:
                    if color[v] >= 0:
                        used.add(color[v])

                c = 0
                while c in used:
                    c += 1
                color[u] = c

                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        next_level.append(v)


            next_level.sort(key=lambda x: len(adj[x]), reverse=True)

            for v in next_level:
                q.append(v)

    return color