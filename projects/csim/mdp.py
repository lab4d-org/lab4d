"""
Markov Decision Processes (Chapter 17)

First we define an MDP, and the special case of a GridMDP, in which
states are laid out in a 2-dimensional grid. We also represent a policy
as a dictionary of {state: action} pairs, and a Utility function as a
dictionary of {state: number} pairs. We then define the value_iteration
and policy_iteration algorithms.
"""

import random
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
import operator
import io
from PIL import Image

orientations = EAST, NORTH, WEST, SOUTH = [(1, 0), (0, 1), (-1, 0), (0, -1)]
turns = LEFT, RIGHT = (+1, -1)


# visualize reward
def visulize_reward(reward_grid, pi):
    # make y axis point upwards
    plt.imshow(reward_grid, cmap="gray", interpolation="nearest")
    plt.gca().invert_yaxis()

    # overlay the policy icon on the reward grid
    for (x, y), action in pi.items():
        if action == (0, 1):
            icon = "^"
        elif action == (0, -1):
            icon = "v"
        elif action == (1, 0):
            icon = ">"
        elif action == (-1, 0):
            icon = "<"
        else:
            icon = "o"
        plt.annotate(
            icon, xy=(x, y), xytext=(x, y), ha="center", va="center", color="red"
        )
    img = get_img_from_plt()
    return img


def value_iteration_instru(mdp, iterations=20):
    U_over_time = []
    U1 = {s: 0 for s in mdp.states}
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for _ in range(iterations):
        U = U1.copy()
        for s in mdp.states:
            U1[s] = R(s) + gamma * max(
                [sum([p * U[s1] for (p, s1) in T(s, a)]) for a in mdp.actions(s)]
            )
        U_over_time.append(U)
    return U_over_time


def sample_trajectory(reward_grid, pi, start_position=(0, 0), max_step=100):
    # evaluate policy and draw the path
    path = []
    path.append(start_position)
    current_position = start_position
    counter = 0
    # while current_position != (3, 2) and current_position != (3, 1):
    while True:
        movement = pi[current_position]
        current_position = (
            current_position[0] + movement[0],
            current_position[1] + movement[1],
        )
        path.append(current_position)
        if (
            current_position[0] < 0
            or current_position[0] > len(reward_grid) - 1
            or current_position[1] < 0
            or current_position[1] > len(reward_grid[0]) - 1
        ):
            break
        counter += 1
        if counter > max_step:
            break
    return path


def plot_trajectory(path):
    for i in range(len(path)):
        # mark start and end
        if i == 0:
            # mark with a green square
            plt.plot(
                path[i][0],
                path[i][1],
                marker="s",
                markersize=10,
                color="green",
                label="start",
            )
        elif i == len(path) - 1:
            # mark with a red circle
            plt.plot(
                path[i][0],
                path[i][1],
                marker="o",
                markersize=10,
                color="red",
                label="end",
            )
        plt.annotate(
            i, xy=path[i], xytext=path[i], ha="center", va="center", color="blue"
        )
        # draw a line from prev_position to current_position
        if i > 0:
            plt.plot(
                [path[i - 1][0], path[i][0]], [path[i - 1][1], path[i][1]], color="blue"
            )
    img = get_img_from_plt()
    return img


def get_img_from_plt():
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)

    # Convert the image to RGB format (if not already in this format)
    img_rgb = img.convert("RGB")

    return np.array(img_rgb)


# visualize the value function updates
def visulize_value_update(sequential_decision_environment, iter_num=100):
    U_over_time = value_iteration_instru(sequential_decision_environment, iter_num)
    plot_grid_step = make_plot_grid_step_function(
        sequential_decision_environment.cols,
        sequential_decision_environment.rows,
        U_over_time,
    )
    frames = []
    for i in range(iter_num):
        plot_grid_step(i)
        img = get_img_from_plt()
        frames.append(img)
        plt.close()
    return frames


def make_plot_grid_step_function(columns, rows, U_over_time):
    """ipywidgets interactive function supports single parameter as input.
    This function creates and return such a function by taking as input
    other parameters."""

    def plot_grid_step(iteration):
        data = U_over_time[iteration]
        data = defaultdict(lambda: 0, data)
        grid = []
        for row in range(rows):
            current_row = []
            for column in range(columns):
                current_row.append(data[(column, row)])
            grid.append(current_row)
        grid.reverse()  # output like book

        figsize_x = len(grid[0]) / 2  # Adjust these ratios to scale the figure size
        figsize_y = len(grid) / 2
        fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))
        ax.imshow(grid, cmap=plt.cm.gray, interpolation="nearest")

        plt.axis("off")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        for col in range(len(grid)):
            for row in range(len(grid[0])):
                magic = grid[col][row]
                ax.text(row, col, "{0:.2f}".format(magic), va="center", ha="center")
        plt.show()

    return plot_grid_step


def isnumber(x):
    """Is x a number?"""
    return hasattr(x, "__int__")


def print_table(table, header=None, sep="   ", numfmt="{}"):
    """Print a list of lists as a table, so that columns line up nicely.
    header, if specified, will be printed as the first row.
    numfmt is the format for all numbers; you might want e.g. '{:.2f}'.
    (If you want different formats in different columns,
    don't use print_table.) sep is the separator between columns."""
    justs = ["rjust" if isnumber(x) else "ljust" for x in table[0]]

    if header:
        table.insert(0, header)

    table = [[numfmt.format(x) if isnumber(x) else x for x in row] for row in table]

    sizes = list(
        map(
            lambda seq: max(map(len, seq)), list(zip(*[map(str, row) for row in table]))
        )
    )

    for row in table:
        print(
            sep.join(
                getattr(str(x), j)(size) for (j, size, x) in zip(justs, sizes, row)
            )
        )


def turn_heading(heading, inc, headings=orientations):
    return headings[(headings.index(heading) + inc) % len(headings)]


def turn_right(heading):
    return turn_heading(heading, RIGHT)


def turn_left(heading):
    return turn_heading(heading, LEFT)


def vector_add(a, b):
    """Component-wise addition of two vectors."""
    return tuple(map(operator.add, a, b))


class MDP:
    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a gamma value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text. Instead of P(s' | s, a) being a probability number for each
    state/state/action triplet, we instead have T(s, a) return a
    list of (p, s') pairs. We also keep track of the possible states,
    terminal states, and actions for each state. [Page 646]"""

    def __init__(
        self,
        init,
        actlist,
        terminals,
        transitions=None,
        reward=None,
        states=None,
        gamma=0.9,
    ):
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")

        # collect states from transitions table if not passed.
        self.states = states or self.get_states_from_transitions(transitions)

        self.init = init

        if isinstance(actlist, list):
            # if actlist is a list, all states have the same actions
            self.actlist = actlist

        elif isinstance(actlist, dict):
            # if actlist is a dict, different actions for each state
            self.actlist = actlist

        self.terminals = terminals
        self.transitions = transitions or {}
        if not self.transitions:
            print("Warning: Transition table is empty.")

        self.gamma = gamma

        self.reward = reward or {s: 0 for s in self.states}

        # self.check_consistency()

    def R(self, state):
        """Return a numeric reward for this state."""

        return self.reward[state]

    def T(self, state, action):
        """Transition model. From a state and an action, return a list
        of (probability, result-state) pairs."""

        if not self.transitions:
            raise ValueError("Transition model is missing")
        else:
            return self.transitions[state][action]

    def actions(self, state):
        """Return a list of actions that can be performed in this state. By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""

        if state in self.terminals:
            return [None]
        else:
            return self.actlist

    def get_states_from_transitions(self, transitions):
        if isinstance(transitions, dict):
            s1 = set(transitions.keys())
            s2 = set(
                tr[1]
                for actions in transitions.values()
                for effects in actions.values()
                for tr in effects
            )
            return s1.union(s2)
        else:
            print("Could not retrieve states from transitions")
            return None

    def check_consistency(self):
        # check that all states in transitions are valid
        assert set(self.states) == self.get_states_from_transitions(self.transitions)

        # check that init is a valid state
        assert self.init in self.states

        # check reward for each state
        assert set(self.reward.keys()) == set(self.states)

        # check that all terminals are valid states
        assert all(t in self.states for t in self.terminals)

        # check that probability distributions for all actions sum to 1
        for s1, actions in self.transitions.items():
            for a in actions.keys():
                s = 0
                for o in actions[a]:
                    s += o[0]
                assert abs(s - 1) < 0.001


class MDP2(MDP):
    """
    Inherits from MDP. Handles terminal states, and transitions to and from terminal states better.
    """

    def __init__(self, init, actlist, terminals, transitions, reward=None, gamma=0.9):
        MDP.__init__(self, init, actlist, terminals, transitions, reward, gamma=gamma)

    def T(self, state, action):
        if action is None:
            return [(0.0, state)]
        else:
            return self.transitions[state][action]


class GridMDP(MDP):
    """A two-dimensional grid MDP, as in [Figure 17.1]. All you have to do is
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state). Also, you should specify the terminal states.
    An action is an (x, y) unit vector; e.g. (1, 0) means move east."""

    def __init__(self, grid, terminals, init=(0, 0), gamma=0.9):
        grid.reverse()  # because we want row 0 on bottom, not on top
        reward = {}
        states = set()
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid
        for x in range(self.cols):
            for y in range(self.rows):
                # if grid[y][x] is not None:
                if grid[y][x]:
                    states.add((x, y))
                    reward[(x, y)] = grid[y][x]
        self.states = states
        actlist = orientations
        transitions = {}
        for s in states:
            transitions[s] = {}
            for a in actlist:
                transitions[s][a] = self.calculate_T(s, a)
        MDP.__init__(
            self,
            init,
            actlist=actlist,
            terminals=terminals,
            transitions=transitions,
            reward=reward,
            states=states,
            gamma=gamma,
        )

    def calculate_T(self, state, action):
        if action:
            return [
                (0.8, self.go(state, action)),
                (0.1, self.go(state, turn_right(action))),
                (0.1, self.go(state, turn_left(action))),
            ]
        else:
            return [(0.0, state)]

    def T(self, state, action):
        return self.transitions[state][action] if action else [(0.0, state)]

    def go(self, state, direction):
        """Return the state that results from going in this direction."""

        state1 = vector_add(state, direction)
        return state1 if state1 in self.states else state

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""

        return list(
            reversed(
                [
                    [mapping.get((x, y), None) for x in range(self.cols)]
                    for y in range(self.rows)
                ]
            )
        )

    def to_arrows(self, policy):
        chars = {(1, 0): ">", (0, 1): "^", (-1, 0): "<", (0, -1): "v", None: "."}
        return self.to_grid({s: chars[a] for (s, a) in policy.items()})


# ______________________________________________________________________________


""" [Figure 17.1]
A 4x3 grid environment that presents the agent with a sequential decision problem.
"""

sequential_decision_environment = GridMDP(
    [[-0.04, -0.04, -0.04, +1], [-0.04, None, -0.04, -1], [-0.04, -0.04, -0.04, -0.04]],
    terminals=[(3, 2), (3, 1)],
)


# ______________________________________________________________________________


def value_iteration(mdp, epsilon=0.001):
    """Solving an MDP by value iteration. [Figure 17.4]"""

    U1 = {s: 0 for s in mdp.states}
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        U = U1.copy()
        delta = 0
        for s in mdp.states:
            U1[s] = R(s) + gamma * max(
                sum(p * U[s1] for (p, s1) in T(s, a)) for a in mdp.actions(s)
            )
            delta = max(delta, abs(U1[s] - U[s]))
        if delta <= epsilon * (1 - gamma) / gamma:
            return U


def best_policy(mdp, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action. [Equation 17.4]"""

    pi = {}
    for s in mdp.states:
        pi[s] = max(mdp.actions(s), key=lambda a: expected_utility(a, s, U, mdp))
    return pi


def expected_utility(a, s, U, mdp):
    """The expected utility of doing a in state s, according to the MDP and U."""

    return sum(p * U[s1] for (p, s1) in mdp.T(s, a))


# ______________________________________________________________________________


def policy_iteration(mdp):
    """Solve an MDP by policy iteration [Figure 17.7]"""

    U = {s: 0 for s in mdp.states}
    pi = {s: random.choice(mdp.actions(s)) for s in mdp.states}
    while True:
        U = policy_evaluation(pi, U, mdp)
        unchanged = True
        for s in mdp.states:
            a = max(mdp.actions(s), key=lambda a: expected_utility(a, s, U, mdp))
            if a != pi[s]:
                pi[s] = a
                unchanged = False
        if unchanged:
            return pi


def policy_evaluation(pi, U, mdp, k=20):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""

    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for i in range(k):
        for s in mdp.states:
            U[s] = R(s) + gamma * sum(p * U[s1] for (p, s1) in T(s, pi[s]))
    return U


class POMDP(MDP):
    """A Partially Observable Markov Decision Process, defined by
    a transition model P(s'|s,a), actions A(s), a reward function R(s),
    and a sensor model P(e|s). We also keep track of a gamma value,
    for use by algorithms. The transition and the sensor models
    are defined as matrices. We also keep track of the possible states
    and actions for each state. [Page 659]."""

    def __init__(
        self,
        actions,
        transitions=None,
        evidences=None,
        rewards=None,
        states=None,
        gamma=0.95,
    ):
        """Initialize variables of the pomdp"""

        if not (0 < gamma <= 1):
            raise ValueError("A POMDP must have 0 < gamma <= 1")

        self.states = states
        self.actions = actions

        # transition model cannot be undefined
        self.t_prob = transitions or {}
        if not self.t_prob:
            print("Warning: Transition model is undefined")

        # sensor model cannot be undefined
        self.e_prob = evidences or {}
        if not self.e_prob:
            print("Warning: Sensor model is undefined")

        self.gamma = gamma
        self.rewards = rewards

    def remove_dominated_plans(self, input_values):
        """
        Remove dominated plans.
        This method finds all the lines contributing to the
        upper surface and removes those which don't.
        """

        values = [val for action in input_values for val in input_values[action]]
        values.sort(key=lambda x: x[0], reverse=True)

        best = [values[0]]
        y1_max = max(val[1] for val in values)
        tgt = values[0]
        prev_b = 0
        prev_ix = 0
        while tgt[1] != y1_max:
            min_b = 1
            min_ix = 0
            for i in range(prev_ix + 1, len(values)):
                if values[i][0] - tgt[0] + tgt[1] - values[i][1] != 0:
                    trans_b = (values[i][0] - tgt[0]) / (
                        values[i][0] - tgt[0] + tgt[1] - values[i][1]
                    )
                    if 0 <= trans_b <= 1 and trans_b > prev_b and trans_b < min_b:
                        min_b = trans_b
                        min_ix = i
            prev_b = min_b
            prev_ix = min_ix
            tgt = values[min_ix]
            best.append(tgt)

        return self.generate_mapping(best, input_values)

    def remove_dominated_plans_fast(self, input_values):
        """
        Remove dominated plans using approximations.
        Resamples the upper boundary at intervals of 100 and
        finds the maximum values at these points.
        """

        values = [val for action in input_values for val in input_values[action]]
        values.sort(key=lambda x: x[0], reverse=True)

        best = []
        sr = 100
        for i in range(sr + 1):
            x = i / float(sr)
            maximum = (values[0][1] - values[0][0]) * x + values[0][0]
            tgt = values[0]
            for value in values:
                val = (value[1] - value[0]) * x + value[0]
                if val > maximum:
                    maximum = val
                    tgt = value

            if all(any(tgt != v) for v in best):
                best.append(np.array(tgt))

        return self.generate_mapping(best, input_values)

    def generate_mapping(self, best, input_values):
        """Generate mappings after removing dominated plans"""

        mapping = defaultdict(list)
        for value in best:
            for action in input_values:
                if any(all(value == v) for v in input_values[action]):
                    mapping[action].append(value)

        return mapping

    def max_difference(self, U1, U2):
        """Find maximum difference between two utility mappings"""

        for k, v in U1.items():
            sum1 = 0
            for element in U1[k]:
                sum1 += sum(element)
            sum2 = 0
            for element in U2[k]:
                sum2 += sum(element)
        return abs(sum1 - sum2)


class Matrix:
    """Matrix operations class"""

    @staticmethod
    def add(A, B):
        """Add two matrices A and B"""

        res = []
        for i in range(len(A)):
            row = []
            for j in range(len(A[0])):
                row.append(A[i][j] + B[i][j])
            res.append(row)
        return res

    @staticmethod
    def scalar_multiply(a, B):
        """Multiply scalar a to matrix B"""

        for i in range(len(B)):
            for j in range(len(B[0])):
                B[i][j] = a * B[i][j]
        return B

    @staticmethod
    def multiply(A, B):
        """Multiply two matrices A and B element-wise"""

        matrix = []
        for i in range(len(B)):
            row = []
            for j in range(len(B[0])):
                row.append(B[i][j] * A[j][i])
            matrix.append(row)

        return matrix

    @staticmethod
    def matmul(A, B):
        """Inner-product of two matrices"""

        return [
            [
                sum(ele_a * ele_b for ele_a, ele_b in zip(row_a, col_b))
                for col_b in list(zip(*B))
            ]
            for row_a in A
        ]

    @staticmethod
    def transpose(A):
        """Transpose a matrix"""

        return [list(i) for i in zip(*A)]


def pomdp_value_iteration(pomdp, epsilon=0.1):
    """Solving a POMDP by value iteration."""

    U = {"": [[0] * len(pomdp.states)]}
    count = 0
    while True:
        count += 1
        prev_U = U
        values = [val for action in U for val in U[action]]
        value_matxs = []
        for i in values:
            for j in values:
                value_matxs.append([i, j])

        U1 = defaultdict(list)
        for action in pomdp.actions:
            for u in value_matxs:
                u1 = Matrix.matmul(
                    Matrix.matmul(
                        pomdp.t_prob[int(action)],
                        Matrix.multiply(pomdp.e_prob[int(action)], Matrix.transpose(u)),
                    ),
                    [[1], [1]],
                )
                u1 = Matrix.add(
                    Matrix.scalar_multiply(pomdp.gamma, Matrix.transpose(u1)),
                    [pomdp.rewards[int(action)]],
                )
                U1[action].append(u1[0])

        U = pomdp.remove_dominated_plans_fast(U1)
        # replace with U = pomdp.remove_dominated_plans(U1) for accurate calculations

        if count > 10:
            if (
                pomdp.max_difference(U, prev_U)
                < epsilon * (1 - pomdp.gamma) / pomdp.gamma
            ):
                return U


if __name__ == "__main__":
    import cv2
    import sys, os

    sys.path.insert(0, os.getcwd())
    from lab4d.utils.io import save_vid

    # reward_grid = np.random.randn(16, 16).tolist()
    reward_grid = [
        [-0.4, -0.4, -0.4, +1],
        [-0.4, 0, -0.4, -1],
        [-0.4, -0.4, -0.4, -0.4],
    ]
    sequential_decision_environment = GridMDP(reward_grid, terminals=[])
    pi = best_policy(
        sequential_decision_environment,
        value_iteration(sequential_decision_environment, 0.001),
    )
    path = sample_trajectory(reward_grid, pi, start_position=(0, 0))

    # visulzie reward, policy and trajectory
    img = visulize_reward(reward_grid, pi)
    cv2.imwrite("tmp/reward.jpg", img[..., ::-1])
    img = plot_trajectory(path)
    cv2.imwrite("tmp/path.jpg", img[..., ::-1])
    print("saved to tmp/reward.jpg")
    print("saved to tmp/path.jpg")

    frames = visulize_value_update(sequential_decision_environment, 100)
    save_vid("tmp/vid", frames)
    print("saved to tmp/vid.mp4")
    print_table(sequential_decision_environment.to_arrows(pi))
