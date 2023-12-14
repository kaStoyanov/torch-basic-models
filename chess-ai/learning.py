import time
import pandas as pd
import matplotlib.pyplot as plt
import random
import chess
import chess.engine
from dqn import DQN
from engine import ChessAgent

def Learning(agent, stockfish, games_to_play, max_game_moves, board_config=None):

    loss = []
    final_score = []
    games = 0
    steps = 0
    start_time = time.time()

    # we play n games
    while games < games_to_play:
        print(games, "games played in", time.time() - start_time, "seconds")
        games += 1

        # Create a new standard board
        if board_config is None:
            board = chess.Board()
        
        # Create a board with the desired configuration (pieces and starting positions)
        else:
            board = chess.Board(board_config)

        done = False
        game_moves = 0

        # analyse board with stockfish
        analysis = stockfish.analyse(board=board, limit=chess.engine.Limit(depth=5))
        
        # get best possible move according to stockfish (with depth=5)
        best_move = analysis['pv'][0]

        # until game is not finished
        while not done:

            game_moves += 1
            steps += 1

            # choose action, here the agent choose whether to explore or exploit
            action_index, move, bit_state, valid_move_tensor = agent.select_action(board, best_move)

            # save this score to compute the reward after the opponent move
            board_score_before = analysis['score'].relative.score(mate_score=10000) / 100

            # white moves
            board.push(move)

            # the game is finished (checkmate, stalemate, draw conditions, ...) or we reached max moves
            done = board.result() != '*' or game_moves > max_game_moves
            
            if done:
                
                final_result = board.result()
                
                # if the game is still not finished (meaning we reached max moves without ending the game) or draw
                # we assign a negative reward
                if final_result == '*' or final_result == "1/2-1/2":
                    reward = -10
                
                # if white wins
                elif final_result == "1-0":
                    reward = 1000
                
                # if black wins
                else:
                    reward = -1000

                # store sample in memory
                agent.remember(agent.MAX_PRIORITY, bit_state, action_index, reward, None, done, valid_move_tensor, None)
                
                board_score_after = reward
                
            # game not finished
            else:

                # black moves
                board.push(random.choice(list(board.legal_moves)))

                # board score is back to our perspective after black moves, so no need to change signs
                analysis = stockfish.analyse(board=board, limit=chess.engine.Limit(depth=5))
                board_score_after = analysis['score'].relative.score(mate_score=10000) / 100

                # is game finished?
                done = board.result() != '*'
                
                # if not done, update next best move
                if not done:
                    best_move = analysis['pv'][0]

                next_bit_state = agent.convert_state(board)
                next_valid_move_tensor, _ = agent.mask_and_valid_moves(board)
                
                # divide by 100 to convert from centipawns to pawns score
                reward = board_score_after - board_score_before - 0.01
                                
                # store sample in memory
                agent.remember(agent.MAX_PRIORITY, bit_state, action_index, reward, next_bit_state, done, valid_move_tensor, next_valid_move_tensor)
                                

            # train model and store loss
            loss.append(agent.learn_experience_replay(debug=False))

            # adjust epsilon (exploration rate)
            agent.adaptiveEGreedy()

        # save final game score
        final_score.append(board_score_after)

    # plot training results
    score_df = pd.DataFrame(final_score, columns=["score"])
    score_df['ma'] = score_df["score"].rolling(window = games // 5).mean()
    loss_df = pd.DataFrame(loss, columns=["loss"])
    loss_df['ma'] = loss_df["loss"].rolling(window=steps // 5).mean()

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Plot the score chart in the first subplot
    ax1.plot(score_df.index, score_df["score"], linewidth=0.2)
    ax1.plot(score_df.index, score_df["ma"])
    ax1.set_title('Final score by game')

    # Plot the loss chart in the second subplot
    ax2.plot(loss_df.index, loss_df["loss"], linewidth=0.1)
    ax2.plot(loss_df.index, loss_df["ma"])
    ax2.set_title('Loss by training step')

    # Show the plot
    plt.show()