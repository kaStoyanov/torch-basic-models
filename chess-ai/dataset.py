from torch.utils.data import Dataset
import os
import torch
import chess
import numpy as np

class CCRLDataset( Dataset ):
    def __init__( self, ccrl_dir ):
        self.ccrl_dir = ccrl_dir
        self.pgn_file_names = os.listdir( ccrl_dir ) # Load Dataset

    def __len__( self ):
        return len( self.pgn_file_names ) # Dataset Length

    def __getitem__( self, idx ):
        """
        Load the game in idx.pgn
        Get a random position, the move made from it, and the winner
        Encode these as numpy arrays
        
        Args:
            idx (int) the index into the dataset.
        
        Returns:
           position (torch.Tensor (16, 8, 8) float32) the encoded position
           policy (torch.Tensor (1) long) the target move's index
           value (torch.Tensor (1) float) the encoded winner of the game
           mask (torch.Tensor (72, 8, 8) int) the legal move mask
        """
        print(idx, '????')
        pgn_file_name = self.pgn_file_names[ idx ] # Get PGN file name from Index
        pgn_file_name = os.path.join( self.ccrl_dir, pgn_file_name ) # Full file name
        with open( pgn_file_name ) as pgn_fh:
            game = chess.pgn.read_game(pgn_fh) # Read PGN file
        moves = list(game.mainline_moves()) # Puts all the moves into a list. E.g. [Move.from_uci('e2e4'), ..., Move.from_uci('c4c8')]
        randIdx = int(np.random.random() * ( len( moves ) - 1 )) # Gets a random move's index from the [moves] list.
        board = game.board() 

        for idx, move in enumerate(moves): # Moves through each chess piece on the board. UNTIL randIdx has been reached.
            board.push(move)
            if (randIdx == idx): # Checks if the random selected move is the current move
                next_move = moves[idx + 1] # Defines the move AFTER randIdx
                break

        winner = encoder.parseResult(game.headers['Result']) # Gets winner of the SELECTED game

        position, policy, value, mask = encoder.encodeTrainingPoint(board, next_move, winner)
        """
        position = the encoded position that the AI can understand -> encodePosition(board)
        value = winner of the game
        mask = a mask containing all legal moves
        """
            
        return { 'position': torch.from_numpy( position ),
                 'policy': torch.Tensor( [policy] ).type( dtype=torch.long ),
                 'value': torch.Tensor( [value] ),
                 'mask': torch.from_numpy( mask )}