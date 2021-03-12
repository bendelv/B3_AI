from pacman_module.game import Agent
from pacman_module.pacman import Directions
import numpy as np
from pacman_module.game import Agent
from pacman_module.util import manhattanDistance


class PacmanAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.initFood = 0
        self.args = args

    def useful_state(self, state):
        """
        arg:
        - State: representation of the game at the time t.

        This method extracts important informations from the state.

        return:
        - The grid of food dots, the pacman position and the ghost position.
        """
        pacmanPos = state.getPacmanPosition()
        ghostPos = state.getGhostPosition(1)

        return state.getFood(), pacmanPos, ghostPos

    def eval(self, state, foodSet):
        """
        arg:
        - State: representation of the game at the time t.
        - foodSet: set containing the positions of the food.

        Evaluation based on minimal manhattan distance between pacman and
        foods, manhattan distance, the number of food eaten and the score.

        return:
        - evalution of a state.
        """
        foodMD = np.Infinity
        pacmanPos = state.getPacmanPosition()
        for elmt in foodSet:
            MD = manhattanDistance(elmt, pacmanPos)
            if MD < foodMD:
                foodMD = MD

        nbFood = self.initFood - state.getNumFood()
        ghostMD = manhattanDistance(pacmanPos, state.getGhostPosition(1))

        evalFood = (1/(foodMD+1))
        evalGhost = (1.0/(ghostMD+1))
        evaluation = state.getScore() + 100*evalFood + 50*nbFood + evalGhost

        if state.isLose():
            evaluation = 0

        return (evaluation, None)

    def cutoff_test(self, state, depth):
        """
        arg:
        - State: representation of the game at the time t.
        - Depth: depth of the recursion.

        Allow to stop recursion at a given depth.

        return:
        - Boolean determining if cutoff state or not.
        """
        depthCut = 3

        if depthCut <= depth or state.isWin() or state.isLose():
            return True
        return False

    def hminimax(self, state, visited, depth):
        """
        arg:
        - State: representation of the game at the time t.
        - Visited: list of visited state.
        - Depth: depth of the recursion.

        return:
        - value: the best next action for pacman.
        """
        foodSet = set()
        foodPos = state.getFood()
        for i in np.arange(foodPos.width):
            for j in np.arange(foodPos.height):
                if foodPos[i][j]:
                    foodSet.add((i, j))
        value = self.max_value(state, visited, depth, foodSet)

        return value

    def max_value(self, state, visited, depth, foodSet):
        """
        arg:
        - State: representation of the game at the time t.
        - Visited: list of visited state.
        - Depth: depth of the recursion.
        - FoodSet: set containing the positions of the food.

        This method is recursive. It represents the pacman play which calls
        recursively the method handling the ghost turn (minSuccessor).

        return:
        - value: the move with the best value for pacman.
        """
        if self.cutoff_test(state, depth):
            return self.eval(state, foodSet)

        value = (-np.Infinity, None)
        successors = state.generatePacmanSuccessors()
        depth = depth + 1

        for successorLoop in successors:
            if self.useful_state(successorLoop[0]) not in visited:
                copyVisited = visited.copy()
                copyVisited.append(self.useful_state(state))

                x, y = successorLoop[0].getPacmanPosition()
                food = state.hasFood(x, y)

                min_val = self.min_value(successorLoop[0], copyVisited,
                                         depth, foodSet)[0]
                tmp = max(value[0], min_val)

                if value[0] < tmp:
                    value = (tmp, successorLoop[1])

        return value

    def min_value(self, state, visited, depth, foodSet):
        """
        arg:
        - State: representation of the game at the time t.
        - Visited: list of visited state.
        - Depth: depth of the recursion.
        - FoodSet: set containing the positions of the food.

        This method is recursive. It represents the ghost play which calls
        recursively the method handling the pacman turn (maxSuccessor).

        return:
        - value: the move with the best value for ghost.
        """
        if self.cutoff_test(state, depth):
            return self.eval(state, foodSet)

        value = (np.Infinity, None)
        successors = state.generateGhostSuccessors(1)
        depth = depth + 1
        for successorLoop in successors:
            max_val = self.max_value(successorLoop[0], visited,
                                     depth, foodSet)[0]
            tmp = min(value[0], max_val)

            if value[0] > tmp:
                value = (tmp, successorLoop[1])

        return value

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
        visited = []
        self.initFood = state.getNumFood()
        return self.hminimax(state, visited, 0)[1]
