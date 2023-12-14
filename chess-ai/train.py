import numpy as np
import torch
import chess.pgn
import chess
from actions import mask_and_valid_moves
from dataset import CCRLDataset
import chess.engine
import random
from engine import ChessAgent
# pgn = open("chess-ai/hikaru_small.pgn")
# pgn = open("chess-ai/Nakamura.pgn")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from learning import Learning
def print_bitboard(bitboard):
    
    # for each row
    for i in range(8):
        
        # print bitboard of 1s/0s (we have to mirror the bitboard)
        if i == 0:
            print(format(bitboard, "064b")[8*(i+1)-1::-1])
        else:
            print(format(bitboard, "064b")[8*(i+1)-1:8*i-1:-1])

import chess
board = chess.Board()
# print_bitboard(board.pawns)
# print_bitboard(board.kings)
# print_bitboard(board.bishops)
mask, valid_moves_dict = mask_and_valid_moves(board)
# print(mask)
print(board)

# for now our agent choose a random move
def agent_choose_move(board):
    return random.choice(list(board.legal_moves))

# Create a chess board
board = chess.Board()

# Create a stockfish engine instance
stockfish = chess.engine.SimpleEngine.popen_uci("chess-ai/stockfish/stockfish-ubuntu-x86-64-avx2")
stockfish
# Analyse starting board with stockfish
board_score_before = stockfish.analyse(board=board, limit=chess.engine.Limit(depth=5))\
    ['score'].relative.score(mate_score=10000)

print(board_score_before)
# Agent choose move
move = agent_choose_move(board)
board.push(move)

# Make random move for black
board.push(random.choice(list(board.legal_moves)))

# Analyse final board with stockfish
board_score_after = stockfish.analyse(board=board, limit=chess.engine.Limit(depth=5))\
    ['score'].relative.score(mate_score=10000)

# Divide by 100 to transform to centipawn to pawn score and subtract 0.01 to penalize the agent for each move. 
# We want to win as fast as possible ;)
reward = board_score_after/100 - board_score_before/100 - 0.01
print(reward)
print(board)
# Exploration rate
epsilon = 1
epsilon_decay = 0.995
epsilon_min = 0.01

# choose random with probability epsilon
if random.uniform(0, 1) <= epsilon:
    move = random.choice(list(board.legal_moves))
    
# choose best move with probability 1 - epsilon
else:
    move = agent_choose_move(board)

# reduce exploration rate after each step
epsilon = max(epsilon * epsilon_decay, epsilon_min)



# Train the model with Experience Replay
def learn_experience_replay(self, debug=False):
    
    batch_size = self.batch_size

    # if memory does not have enough sample to fill a batch, return
    if len(self.memory) < batch_size:
        return

    # get priorities from the first element in the memory samples tuple
    priorities = [x[0] for x in self.memory]
    
    # the higher the priority, the more probable the sample will be included in the batch training
    priorities_tot = np.sum(priorities)
    weights = priorities / priorities_tot

    # extract samples for the batch training
    minibatch_indexes = np.random.choice(range(len(self.memory)), size=batch_size, replace=False, p=weights)
    minibatch = [self.memory[x] for x in minibatch_indexes]

    # unpack the tuples in the batch into different lists, to be converted into tensors
    state_list = []
    state_valid_moves = []
    action_list = []
    reward_list = []
    next_state_list = []
    next_state_valid_moves = []
    done_list = []

    for priority, bit_state, action, reward, next_bit_state, done, state_valid_move, next_state_valid_move in minibatch:

        # bit state is the 16 bitboards of the state before the move
        state_list.append(bit_state)
        
        # state_valid_moves is a tensor containing the indexes of valid moves (out of 4096)
        state_valid_moves.append(state_valid_move.unsqueeze(0))
        
        # action is the index of the chosen move (out of 4096)
        action_list.append([action])
        
        # reward is the reward obtained by making the chosen move
        reward_list.append(reward)
        
        # done indicates if the game ended after making the chosen move
        done_list.append(done)

        if not done:
            
            # next_bit_state is the 16 bitboards of the state after the move
            next_state_list.append(next_bit_state)
            
            # next_state_valid_moves is a tensor containing the indexes of valid moves (out of 4096)
            next_state_valid_moves.append(next_state_valid_move.unsqueeze(0))

    # state_valid_moves and next_state_valid_moves are already tensors, we just need to concat them
    state_valid_move_tensor = torch.cat(state_valid_moves, 0)
    next_state_valid_move_tensor = torch.cat(next_state_valid_moves, 0)

    # convert all lists to tensors
    state_tensor = torch.from_numpy(np.array(state_list)).float()
    action_list_tensor = torch.from_numpy(np.array(action_list, dtype=np.int64))
    reward_list_tensor = torch.from_numpy(np.array(reward_list)).float()
    next_state_tensor = torch.from_numpy(np.array(next_state_list)).float()
    
    # create a tensor with 
    bool_array = np.array([not x for x in done_list])
    not_done_mask = torch.tensor(bool_array, dtype=torch.bool)
    
    # compute the expected rewards for each valid move
    policy_action_values = self.policy_net(state_tensor, state_valid_move_tensor)
    
    # get only the expected reward for the chosen move (to calculate loss against the actual reward)
    policy_action_values = policy_action_values.gather(1, action_list_tensor)
    
    # target values are what we want the network to predict (our actual values in the loss function)
    # target values = reward + max_reward_in_next_state * gamma
    # gamma is the discount factor and tells the agent whether to prefer long term rewards or immediate rewards. 0 = greedy, 1 = long term 
    max_reward_in_next_state = torch.zeros(batch_size, dtype=torch.double)
    
    with torch.no_grad():
        
        # if the state is final (done = True, not_done_mask = False) the max_reward_in_next_state stays 0 
        max_reward_in_next_state[not_done_mask] = self.policy_net(next_state_tensor, next_state_valid_move_tensor).max(1)[0]
    
    target_action_values = (max_reward_in_next_state * self.gamma) + reward_list_tensor
    target_action_values = target_action_values.unsqueeze(1)
    
    # loss is computed between expected values (predicted) and target values (actual)
    loss = self.loss_function(policy_action_values, target_action_values)

    # Update priorities of samples in memory based on size of error (higher error = higher priority)
    for i in range(batch_size):
        
        predicted_value = policy_action_values[i]
        target_value = target_action_values[i]
        
        # priority = mean squared error
        priority = F.mse_loss(predicted_value, target_value, reduction='mean').detach().numpy()
        
        # change priority of sample in memory
        sample = list(self.memory[minibatch_indexes[i]])
        sample[0] = priority
        self.memory[minibatch_indexes[i]] = tuple(sample)

    # clear gradients of all parameters from the previous training step
    self.optimizer.zero_grad()
    
    # calculate the new gradients of the loss with respect to all the model parameters by traversing the network backwards
    loss.backward()
    
    # adjust model parameters (weights, biases) according to computed gradients and learning rate
    self.optimizer.step()
    
    if debug:
        print("state_tensor shape", state_tensor.shape)
        print("\naction_list_tensor shape", action_list_tensor.shape)
        print("\naction_list_tensor (chosen move out of 4096)", action_list_tensor)
        print("\npolicy_action_values (expected reward of chosen move)", policy_action_values)
        print("\nnot_done_mask", not_done_mask)
        print("\ntarget_action_values", target_action_values)
        print("\nreward_list_tensor", reward_list_tensor)
        print("\nloss:", loss)

    # return loss so that we can plot loss by training step
    return float(loss)

# add this new method to our ChessAgent class
setattr(ChessAgent, "learn_experience_replay", learn_experience_replay)

# generate a random training sample
def generate_random_sample(agent, stockfish, board):
    
    # set a standard priority
    priority = 1
    
    # convert board in 16 bitboards
    state = agent.convert_state(board)
    
    # get valid moves tensor
    valid_moves, _ = agent.mask_and_valid_moves(board)
    
    # choose random move and compute its index (out of 4096)
    random_move = random.choice(list(board.legal_moves))
    action = agent.get_move_index(random_move)
    
    # make random move for white and black and compute reward
    board_score_before = stockfish.analyse(board=board, limit=chess.engine.Limit(depth=5))\
        ['score'].relative.score(mate_score=10000)
    
    board.push(random_move)
    board.push(random.choice(list(board.legal_moves)))
    
    board_score_after = stockfish.analyse(board=board, limit=chess.engine.Limit(depth=5))\
        ['score'].relative.score(mate_score=10000)
    
    # divide by 100 to convert from centipawns to pawns score
    reward = board_score_after / 100 - board_score_before / 100 - 0.01
    
    # convert board in 16 bitboard
    next_state = agent.convert_state(board)
    
    # if board.result() == * the game is not finished
    done = board.result() != '*'
    
    # get valid moves tensor
    next_valid_moves, _ = agent.mask_and_valid_moves(board)
    
    # undo white and black moves
    board.pop()
    board.pop()
    
    # store in agent memory
    agent.remember(priority, state, action, reward, next_state, done, valid_moves, next_valid_moves)

# Create a chess board
board = chess.Board()

# Create an agent
agent = ChessAgent()

# for i in range(16):
    # generate_random_sample(agent, stockfish, board)

# print(len(agent.memory))

# import matplotlib.pyplot as plt

# loss = []
# for i in range(30):
    # loss.append(agent.learn_experience_replay(debug=False))

# plt.plot(loss)
# plt.show()

Learning(agent,stockfish,games_to_play=15,max_game_moves=50)