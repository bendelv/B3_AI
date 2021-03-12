# Complete this class for all parts of the project
import numpy as np
from pacman_module.game import Agent
from pacman_module.pacman import Directions
from pacman_module.pacman import GameState


class PacmanAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args
        self.visited = {}

    def utility(self, state):
        """
        arg:
        - State: representation of the game at the time t.

        return:
        - The score of terminal node.
        """
        return (state.getScore(), None)

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

    def alpha_beta_search(self, state):
        """
        arg:
        - State: representation of the game at the time t.

        return:
        - value: the best next action for pacman.
        """
        value = self.max_value(state, - np.Infinity, np.Infinity)
        return value

    def max_value(self, state, alpha, beta):
        """
        arg:
        - State: representation of the game at the time t.
        - alpha: the lower bound value of visited nodes.
        - beta: the upper bound value of visited nodes.

        This method is recursive. It represents the pacman play which calls
        recursively the method handling the ghost turn (minSuccessor).

        return:
        - value: the move with the best value for pacman.
        """
        if state.isWin() or state.isLose():
            return self.utility(state)

        value = (-np.Infinity, None)
        successors = state.generatePacmanSuccessors()

        for successorLoop in successors:
            usefulState = self.useful_state(successorLoop[0])

            if (self.useful_state(successorLoop[0]) not in self.visited or
               self.visited[usefulState] <= successorLoop[0].getScore()):
                self.visited[usefulState] = successorLoop[0].getScore()
                min_val = self.min_value(successorLoop[0], alpha, beta)[0]
                tmp = max(value[0], min_val)
                if tmp > value[0]:
                    value = (tmp, successorLoop[1])

            else:
                return (- np.Infinity, None)

            if value[0] >= beta:
                return value
            alpha = max(alpha, value[0])

        return value

    def min_value(self, state, alpha, beta):
        """
        arg:
        - State: representation of the game at the time t.
        - alpha: the lower bound value of visited nodes.
        - beta: the upper bound value of visited nodes.

        This method is recursive. It represents the ghost play which calls
        recursively the method handling the pacman turn (maxSuccessor).

        return:
        - value: the move with the best value for ghost.
        """
        if state.isWin() or state.isLose():
            return self.utility(state)

        value = (np.Infinity, None)
        successors = state.generateGhostSuccessors(1)

        for successorLoop in successors:
            max_val = self.max_value(successorLoop[0], alpha, beta)[0]
            tmp = min(value[0], max_val)
            if tmp < value[0]:
                value = (tmp, successorLoop[1])

            if value[0] <= alpha:
                return value
            beta = min(beta, value[0])

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
        self.visited = {}
        score, action = self.alpha_beta_search(state)
        return action
