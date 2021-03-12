# Complete this class for all parts of the project
from pacman_module.game import Agent
from pacman_module.pacman import Directions, GhostRules
import numpy as np
# import matplotlib.pyplot as plt
from pacman_module import util
import sys


class BeliefStateAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args
        """
            Variables to use in 'updateAndFetBeliefStates' method.
            Initialization occurs in 'get_action' method.
        """
        # Current list of belief states over ghost positions
        self.beliefGhostStates = None
        # Grid of walls (assigned with 'state.getWalls()' method)
        self.walls = None
        # Uniform distribution size parameter 'w'
        # for sensor noise (see instructions)
        self.w = self.args.w
        # Probability for 'leftturn' ghost to take 'EAST' action
        # when 'EAST' is legal (see instructions)
        self.p = self.args.p

        self.m = None
        self.n = None
        self.board = []
        self.transitionMatrix = None
        self.sensorMatrix = None

        # self.entropy = []
        # self.iter = 50

    def createTransitionMatrix(self):
        """
        This function calculates probabilities for a ghost to go from the
        position i to the position j, knowing its behaviour.
        Both i and j belongs to all possible positions on the board.

        Return:
            A transition matrix in which we find all the probabilities of
            legal actions for each possible position of the board.
        """
        transitionMatrix = np.zeros((1, self.n*self.m))

        for i in self.board:
            x1, y1 = i
            line = []

            if self.walls[x1][y1] is True:
                line = np.zeros((1, self.n*self.m))
                tmp = np.vstack((transitionMatrix, line.reshape((1, -1))))
                transitionMatrix = tmp
                continue

            count = 0
            left = self.walls[x1 - 1][y1]
            right = self.walls[x1 + 1][y1]
            down = self.walls[x1][y1 - 1]
            up = self.walls[x1][y1 + 1]
            list = [left, right, down, up]
            for near in list:
                if near is False:
                    count = count + 1

            p = self.p

            if count == 0:
                line = np.zeros((1, self.n*self.m))
                tmp = np.vstack((transitionMatrix, line.reshape((1, -1))))
                transitionMatrix = tmp
                continue

            cmp = (1 - p)/count

            for j in self.board:
                x2, y2 = j
                walls = self.walls[x2][y2] is False

                if self.walls[x1 + 1][y1] is False:
                    """
                    If East is a legal action
                    """
                    if x2 == x1 + 1 and y2 == y1:
                        line.append(p + cmp)

                    elif x2 == x1 - 1 and y2 == y1 and walls:
                        line.append(cmp)

                    elif x2 == x1 and y2 == y1 + 1 and walls:
                        line.append(cmp)

                    elif x2 == x1 and y2 == y1 - 1 and walls:
                        line.append(cmp)

                    else:
                        line.append(0)

                else:
                    """
                    If East is not a legal action
                    """
                    if x2 == x1 - 1 and y2 == y1 and walls:
                        line.append(1/count)

                    elif x2 == x1 and y2 == y1 + 1 and walls:
                        line.append(1/count)

                    elif x2 == x1 and y2 == y1 - 1 and walls:
                        line.append(1/count)

                    else:
                        line.append(0)

            line = np.array(line)
            tmp = np.vstack((transitionMatrix, line.reshape((1, -1))))
            transitionMatrix = tmp

        return transitionMatrix[1:, :]

    def createSensorModel(self):
        """
        This function creates a matrix representing a sensor model, containing
        probabilities for a position j to be given as next evidence, depending
        of i, the real position of the ghost.
        Both i and j belongs to all possible positions on the board.
        
        Return:
            A sensor matrix resuming the information of the sensor.

        """
        sensorMatrix = np.zeros((1, self.n*self.m))
        w = self.w
        w2 = 2*w+1
        div = float(w2 * w2)

        for i in self.board:
            x1, y1 = i
            line = []
            for j in self.board:
                x2, y2 = j

                cond1 = (x2 >= x1 - w and x2 <= x1 + w)
                cond2 = (y2 >= y1 - w and y2 <= y1 + w)

                if (cond1 and cond2):
                    line.append(1/div)
                else:
                    line.append(0)

            sensorMatrix = np.vstack((sensorMatrix, line))

        return sensorMatrix[1:, :]

    # def entropyF(self, beliefStates):
    #      """
    #      This function permit to calculate the entropy to can study
    #      the convergence of our belief state
    #
    #      Return:
    #         The entropy of the belief state
    #      """
    #     shape = beliefStates.shape
    #     x = np.ma.log(beliefStates)
    #     x = x.filled(0)
    #     log_belief = x/np.log(2)
    #     return -np.sum(beliefStates*log_belief)

    def updateAndGetBeliefStates(self, evidences):
        """
        Given a list of (noised) distances from pacman to ghosts,
        returns a list of belief states about ghosts positions

        Arguments:
        ----------
        - `evidences`: list of (noised) ghost positions at state x_{t}
          where 't' is the current time step

        Return:
        -------
        - A list of Z belief states at state x_{t} about ghost positions
          as N*M numpy matrices of probabilities
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.

        N.B. : [0,0] is the bottom left corner of the maze
        """
        # XXX: Your code here

        # if self.iter < 0:
        #     np.save('Entropy{}_{}'.format(self.w, self.p), self.entropy)
        #     sys.exit()
        #
        # self.iter = self.iter - 1

        if (self.m or self.n) is None:
            self.m = self.walls.height
            self.n = self.walls.width

        if not self.board:
            for x in np.arange(self.n):
                for y in np.arange(self.m):
                    self.board.append((x, y))

        if self.transitionMatrix is None:
            self.transitionMatrix = self.createTransitionMatrix()

        if self.sensorMatrix is None:
            self.sensorMatrix = self.createSensorModel()

        beliefStates = self.beliefGhostStates

        # self.entropy.append(self.entropyF(beliefStates))

        for i, e in enumerate(evidences):
            """
            To manage multiple ghosts.
            """
            col_beliefStates = np.reshape(beliefStates[i, :, :], (-1, 1))

            index = self.board.index(e)
            O_col = self.sensorMatrix[:, index]

            O = np.diag(O_col)
            """
            O = Observation matrix.
            """

            col_bel = np.dot(O, self.transitionMatrix)
            col_beliefStates = np.dot(col_bel, col_beliefStates)

            alpha = 1/(np.sum(col_beliefStates))
            col_beliefStates = alpha*col_beliefStates

            beliefState = col_beliefStates.reshape((self.n, self.m))
            beliefStates[i, :, :] = beliefState

        # XXX: End of your code
        self.beliefGhostStates = beliefStates
        return beliefStates

    def _computeNoisyPositions(self, state):
        """
            Compute a noisy position from true ghosts positions.
            XXX: DO NOT MODIFY THAT FUNCTION !!!
            Doing so will result in a 0 grade.
        """
        positions = state.getGhostPositions()
        w = self.args.w
        w2 = 2*w+1
        div = float(w2 * w2)
        new_positions = []
        for p in positions:
            (x, y) = p
            dist = util.Counter()
            for i in range(x - w, x + w + 1):
                for j in range(y - w, y + w + 1):
                    dist[(i, j)] = 1.0 / div
            dist.normalize()
            new_positions.append(util.chooseFromDistribution(dist))
        return new_positions

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

        """
           XXX: DO NOT MODIFY THAT FUNCTION !!!
                Doing so will result in a 0 grade.
        """

        # XXX : You shouldn't care on what is going on below.
        # Variables are specified in constructor.
        if self.beliefGhostStates is None:
            self.beliefGhostStates = state.getGhostBeliefStates()
        if self.walls is None:
            self.walls = state.getWalls()
        return self.updateAndGetBeliefStates(
            self._computeNoisyPositions(state))
