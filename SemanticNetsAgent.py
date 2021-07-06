

class SemanticNetsAgent:
    def __init__(self):
        #If you want to do any initial processing, add it here.
        self.moves = [(0,1), (1,0), (1,1), (2,0), (0,2)]

    def legal_move(self, move, node):
        ds, dw = move
        s, w, b = node
        if b:
            return s-ds >= 0 and w-dw >= 0
        else:
            return s+ds <= self.sheeps and w+dw <= self.wolves

    def backtrack(self, pars, s, t):
        path = [t]
        while path[-1] != s:
            path.append(pars[path[-1]])
        path.reverse()
        moves = []
        for a, b in zip(path[1:], path[:-1]):
            moves.append((abs(a[0]-b[0]), abs(a[1]-b[1])))
        return moves

    def solve(self, initial_sheep, initial_wolves):
        self.sheeps = initial_sheep
        self.wolves = initial_wolves
        start = (initial_sheep, initial_wolves, True)
        goal = (0, 0)
        visited = set()
        parent = {}
        queue = []
        queue.append(start)

        while queue:
            n = len(queue)
            for i in range(n):
                cur = queue.pop(0)
                if cur in visited:
                    continue
                else:
                    visited.add(cur)

                left_sheep, left_wolf, boat = cur
                if 0 < left_sheep < left_wolf or 0 < self.sheeps - left_sheep < self.wolves - left_wolf: # check if state is legal
                    continue
                for sd, wd in self.moves:
                    if self.legal_move((sd, wd), cur):
                        if boat:
                            new_sheep = left_sheep - sd
                            new_wolf  = left_wolf - wd
                        else:
                            new_sheep = left_sheep + sd
                            new_wolf = left_wolf + wd

                        new_state = (new_sheep, new_wolf, not boat)

                        if (new_sheep, new_wolf) == goal:
                            parent[(new_state)] = cur
                            return self.backtrack(parent, start, new_state)

                        if new_state not in visited:
                            parent[(new_state)] = cur
                            queue.append(new_state)

        return []


if __name__ == "__main__":

    test_agent = SemanticNetsAgent()

    print(test_agent.solve(1, 1))
    print(test_agent.solve(2, 2))
    print(test_agent.solve(3, 3))
    print(test_agent.solve(5, 3))
    print(test_agent.solve(6, 3))
    print(test_agent.solve(7, 3))
    print(test_agent.solve(5, 5))