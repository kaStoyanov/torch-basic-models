import numpy as np



def mask_and_valid_moves(board):

    mask = np.zeros((64, 64))
    valid_moves_dict = {}
    
    # for each valid move
    for move in board.legal_moves:
        
        # mask is a matrix
        mask[move.from_square, move.to_square] = 1
        
        # compute index based on starting square and target square
        index = 64 * (move.from_square) + (move.to_square)
        
        valid_moves_dict[index] = move
    
    return mask, valid_moves_dict