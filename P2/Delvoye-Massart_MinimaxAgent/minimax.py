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

    def maxSuccessor(self, state):
        """
        arg:
        - State: representation of the game at the time t.

        This method is recursive. It represents the pacman play which calls
        recursively the method handling the terminal state or the player.

        return:
        - maxUtility: the move with the best value for pacman.
        """
        successors = state.generatePacmanSuccessors()
        maxUtility = (- np.Infinity, None)

        for successorLoop in successors:
            usefulState = self.useful_state(successorLoop[0])

            if ((usefulState not in self.visited) or
               self.visited[usefulState] <= successorLoop[0].getScore()):
                    self.visited[usefulState] = successorLoop[0].getScore()
                    scoreUtility = self.minimax(successorLoop[0], False)
                    if scoreUtility[0] > maxUtility[0]:
                        maxUtility = (scoreUtility[0], successorLoop[1])
            else:
                return (- np.Infinity, None)
        return maxUtility

    def minSuccessor(self, state):
        """
        arg:
        - State: representation of the game at the time t.

        This method is recursive. It represents the ghost play which calls
        recursively the method handling the terminal state or the player.

        return:
        - minUtility: the move with the best value for ghost.
        """
        successors = state.generateGhostSuccessors(1)
        minUtility = (np.Infinity, None)

        for successorLoop in successors:
            scoreUtility = self.minimax(successorLoop[0], True)
            if scoreUtility[0] < minUtility[0]:
                minUtility = (scoreUtility[0], successorLoop[1])

        return minUtility

    def minimax(self, state, player):
        """
        arg:
        - State: representation of the game at the time t.
        - Player: Determine if it is pacman (True) or ghost (False) turn.

        We test if it is a terminal state and if not, call the next player
        method.

        return:
        - The utility value if terminal state.
        """
        if state.isWin() or state.isLose():
            return self.utility(state)

        elif player is True:
            return self.maxSuccessor(state)

        else:
            return self.minSuccessor(state)

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
        return self.minimax(state, True)[1]
