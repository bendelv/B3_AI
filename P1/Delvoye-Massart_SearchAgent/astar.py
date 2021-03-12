from pacman_module.game import Agent
from pacman_module.game import Grid
from pacman_module.pacman import Directions
from pacman_module.util import Stack
from pacman_module.util import PriorityQueue
from pacman_module.pacman import GameState
from pacman_module.util import manhattanDistance
from random import randint
import numpy as np


class PacmanAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args
        self.way = Stack()

    def cost_way(self, food, pacmanPos, foodSet, prevCost):
        """
        arg:
        -Food: boolean argument, true if there is food.
        -PacmanPos: position of pacman
        -FoodSet: set containing the position of the food dots
        -PrevCost: contain the previous cost.

        Calculate the cost as the sum of the largest Manhattan distance
        and the cost of the edge.

        return:
        -The global cost of the edge.
        """
        cost = prevCost
        maxMD = 0
        for elmt in foodSet:
            MD = manhattanDistance(elmt, pacmanPos)
            if MD > maxMD:
                maxMD = MD

        if food:
            cost = cost + maxMD + 1
        else:
            cost = cost + maxMD + 10

        return cost

    def useful_state(self, state):
        """
        arg:
        - State: representation of the game at the time t.

        This method extracts important informations from the state.

        return:
        - The grid of food dots and the pacman position.
        """
        return state.getFood(), state.getPacmanPosition()

    def find_goal(self, state):
        """
        arg:
        - State: representation of the game at the time t.

        Define a way to solve the problem (eat all the food).

        return:
        - A stack containing the movements.
        """
        initState = self.useful_state(state)
        fringe = PriorityQueue()
        foodSet = set()
        ways = {}
        priority = 0

        """
        Loop trough the food grid and initialize the set of food positions.
        """
        foodPos = state.getFood()
        for i in np.arange(foodPos.width):
            for j in np.arange(foodPos.height):
                if foodPos[i][j]:
                    foodSet.add((i, j))

        popped = [None, None, None, foodSet]

        while not state.isWin():
            successors = state.generatePacmanSuccessors()

            for succ in successors:
                """
                Get successors after legal actions of current state.
                Push new state, move, current state on the fringe
                and the current set of food positions
                plus the cost of the edge to this successor.
                """
                succState = succ[0]
                succMove = succ[1]

                x, y = succState.getPacmanPosition()
                pacmanPos = [x, y]
                food = state.hasFood(x, y)

                foodSetCopy = popped[3].copy()

                if food:
                    foodSetCopy.remove((x, y))

                cost = self.cost_way(food, pacmanPos, foodSetCopy, priority)
                fringe.push([succState, succMove, state, foodSetCopy], cost)

            while True:
                """
                Pop a new element from the fringe
                as long as it has been visited.
                """
                priority, popped = fringe.pop()
                represCurrState = self.useful_state(popped[0])
                represPrevState = self.useful_state(popped[2])

                if represCurrState not in ways:
                    break

            ways[represCurrState] = [represPrevState, popped[1]]
            state = popped[0]

        moves = Stack()
        key = self.useful_state(popped[0])

        while not key == initState:
            """
            Traceback the way to the initial state and store it.
            """
            moves.push((ways.get(key))[1])
            key = (ways.get(key))[0]

        return moves

    def get_action(self, state):
        """
        Given a pacman game state, returns a legal move.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.

        Return:
        -------
        - A legal move as defined in `game.Directions`.

        """
        if self.way.isEmpty():
            self.way = self.find_goal(state)

        return self.way.pop()
